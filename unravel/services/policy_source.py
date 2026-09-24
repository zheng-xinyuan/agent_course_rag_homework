"""Download Shanghai law records from the official public law-book API."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://law.sfj.sh.gov.cn/yidianApi/api/v1"
WEB_ROOT = "https://law.sfj.sh.gov.cn"
DEFAULT_SOURCE_PATH = Path.home() / ".unravel" / "policy-source" / "shanghai-laws.json"
USER_AGENT = "Unravel-classroom-RAG/1.0 (public law corpus download)"


class PolicySourceError(RuntimeError):
    """Raised when the official law API returns incomplete or invalid data."""


class _LawTextParser(HTMLParser):
    """Extract readable law text while keeping paragraph and table boundaries."""

    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        raw_text = "".join(self.parts).replace("\xa0", " ")
        lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in raw_text.splitlines()]
        return "\n".join(line for line in lines if line)


class _RequestPacer:
    """Limit request starts so a classroom download stays gentle on the source."""

    def __init__(self, minimum_interval: float) -> None:
        self.minimum_interval = max(0.0, minimum_interval)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self.minimum_interval
        if delay:
            time.sleep(delay)


def _fetch_json(url: str, *, timeout: int = 45, retries: int = 4) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or payload.get("code") != 200:
                raise PolicySourceError(f"Unexpected response from official API: {url}")
            return payload
        except HTTPError as exc:
            if exc.code not in {408, 425, 429, 500, 502, 503, 504} or attempt == retries - 1:
                raise PolicySourceError(f"Official API returned HTTP {exc.code}: {url}") from exc
        except (TimeoutError, URLError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise PolicySourceError(f"Failed to fetch official API response: {url}") from exc
        time.sleep(min(2**attempt, 12))
    raise PolicySourceError(f"Failed to fetch official API response: {url}")


def _fetch_summaries(page_size: int = 2000) -> tuple[list[dict[str, Any]], int]:
    query = urlencode({"page": 1, "page_size": page_size})
    payload = _fetch_json(f"{API_ROOT}/lawsearch?{query}")
    rows = payload.get("data")
    pager = payload.get("pager") or {}
    total = int(pager.get("total", 0))
    if not isinstance(rows, list) or total <= 0:
        raise PolicySourceError("The official API did not return a law list and total count")

    summaries = [row for row in rows if isinstance(row, dict) and row.get("law_id")]
    unique = {str(row["law_id"]): row for row in summaries}
    if len(unique) != total:
        raise PolicySourceError(
            f"The official API reports {total} laws but returned {len(unique)} unique IDs; "
            "retry the download later."
        )
    return list(unique.values()), total


def _category_names(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    names: list[str] = []
    pending = list(value)
    while pending:
        item = pending.pop(0)
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
        elif isinstance(item, list):
            pending[0:0] = item
    return " / ".join(names)


def _body_from_detail(detail: dict[str, Any]) -> str:
    rich_text = detail.get("richtext_content")
    if isinstance(rich_text, str) and rich_text.strip():
        parser = _LawTextParser()
        parser.feed(rich_text)
        return parser.text()

    sections = detail.get("laws") or []
    if isinstance(sections, list):
        return "\n".join(
            str(section.get("text") or "").strip()
            for section in sections
            if isinstance(section, dict) and section.get("text")
        )
    return ""


def _map_record(summary: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    law_id = str(detail.get("law_id") or summary["law_id"])
    law_type = str(detail.get("law_type") or summary.get("law_type") or "")
    title = str(detail.get("law_name") or summary.get("law_name") or "")
    return {
        "政策ID": law_id,
        "政策名称": title,
        "发布日期": str(detail.get("release_date") or ""),
        "政策文号": str(detail.get("issuing_number") or ""),
        "政策类别": _category_names(detail.get("category")) or law_type,
        "发布单位": str(detail.get("department") or ""),
        "效力级别": law_type,
        "时效性": str(detail.get("timeliness") or summary.get("timeliness") or ""),
        "政策正文": _body_from_detail(detail),
        "政策链接": f"{WEB_ROOT}/#/detail?id={quote(law_id, safe='')}",
        "实施日期": str(detail.get("implement_date") or summary.get("implement_date") or ""),
        "数据来源": WEB_ROOT,
    }


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PolicySourceError(f"Invalid checkpoint line {line_number}: {path}") from exc
            law_id = str(record.get("政策ID") or "")
            if law_id:
                completed[law_id] = record
    return completed


def download_official_policy_corpus(
    output_path: str | Path = DEFAULT_SOURCE_PATH,
    *,
    workers: int = 12,
    minimum_interval: float = 0.05,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Fetch every public record and full text, resuming from a JSONL checkpoint."""
    if workers < 1 or workers > 12:
        raise ValueError("workers must be between 1 and 12")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.jsonl")
    summaries, total = _fetch_summaries()
    completed = _load_checkpoint(checkpoint)
    summary_by_id = {str(row["law_id"]): row for row in summaries}
    completed = {law_id: record for law_id, record in completed.items() if law_id in summary_by_id}
    pacer = _RequestPacer(minimum_interval)

    pending = [
        (law_id, summary)
        for law_id, summary in summary_by_id.items()
        if law_id not in completed
    ]
    print(
        f"Official Shanghai law corpus: {total} records; "
        f"resuming {len(completed)} cached details; fetching {len(pending)}.",
        flush=True,
    )

    if pending:
        with checkpoint.open("a", encoding="utf-8") as checkpoint_stream:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                finished = 0
                reported_progress = len(completed) // progress_every
                for batch_start in range(0, len(pending), workers):
                    batch = pending[batch_start : batch_start + workers]
                    futures = {}
                    for law_id, summary in batch:
                        pacer.wait()
                        future = executor.submit(
                            _fetch_json,
                            f"{API_ROOT}/lawsearch/{quote(law_id, safe='')}",
                        )
                        futures[future] = (law_id, summary)

                    for future in as_completed(futures):
                        law_id, summary = futures[future]
                        payload = future.result()
                        detail = payload.get("data")
                        if not isinstance(detail, dict):
                            raise PolicySourceError(f"Missing detail for law ID {law_id}")
                        record = _map_record(summary, detail)
                        completed[law_id] = record
                        checkpoint_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        checkpoint_stream.flush()
                        finished += 1

                    progress_bucket = len(completed) // progress_every
                    if (
                        progress_bucket > reported_progress
                        or finished == len(pending)
                    ):
                        print(f"Fetched details: {len(completed)}/{total}", flush=True)
                        reported_progress = progress_bucket

    missing = sorted(set(summary_by_id) - set(completed))
    if missing:
        raise PolicySourceError(
            f"Downloaded {len(completed)}/{total} records; checkpoint retained at {checkpoint}"
        )

    records = [completed[law_id] for law_id in summary_by_id]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    checkpoint.unlink(missing_ok=True)

    status_counts: dict[str, int] = {}
    empty_body_count = 0
    for record in records:
        status = record["时效性"] or "(missing)"
        status_counts[status] = status_counts.get(status, 0) + 1
        if not record["政策正文"]:
            empty_body_count += 1

    return {
        "source_count": total,
        "downloaded_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "empty_body_count": empty_body_count,
        "output_path": str(output),
    }
