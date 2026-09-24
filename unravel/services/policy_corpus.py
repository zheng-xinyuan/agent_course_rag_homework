"""Loading, filtering, and chunking the Shanghai policy corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PPT_EXPECTED_COUNTS = {"active": 1345, "repealed": 410, "superseded": 92}


class PolicyCorpusError(ValueError):
    """Raised when the source corpus does not match the expected schema."""


@dataclass
class PolicyCorpus:
    """Filtered policy records and an auditable status summary."""

    records: list[dict[str, Any]]
    source_count: int
    status_counts: dict[str, int]
    category_counts: dict[str, int]
    unknown_statuses: dict[str, int]


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("data", "records", "policies", "法规", "法规列表")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
    else:
        rows = None

    if rows is None or not all(isinstance(row, dict) for row in rows):
        raise PolicyCorpusError("Expected a JSON array of policy objects")
    return rows


def _classify_status(value: Any) -> str:
    status = re.sub(r"\s+", "", str(value or ""))
    if "失效" in status or "废止" in status:
        return "repealed"
    if "修改" in status and "重新发布" in status:
        return "superseded"
    if status in {"有效", "现行有效", "有效(现行)", "有效（现行）", "仍然有效"}:
        return "active"
    return "unknown"


def load_policy_corpus(source_path: str | Path) -> PolicyCorpus:
    """Load records and keep only current effective laws.

    PPT's final demo excludes both repealed laws and the superseded
    "amended and reissued" editions, leaving 1,345 current laws.
    """
    path = Path(source_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise PolicyCorpusError(f"Cannot read source JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyCorpusError(f"Invalid JSON in {path}: {exc}") from exc

    rows = _records_from_payload(payload)
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    unknown_statuses: Counter[str] = Counter()
    active_records: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=1):
        status = str(row.get("时效性", "")).strip()
        if not status:
            raise PolicyCorpusError(f"Record {row_number} has no 时效性 value")
        status_counts[status] += 1
        category = _classify_status(status)
        category_counts[category] += 1
        if category == "active":
            active_records.append(row)
        elif category == "unknown":
            unknown_statuses[status] += 1

    if unknown_statuses:
        summary = ", ".join(
            f"{status}: {count}" for status, count in sorted(unknown_statuses.items())
        )
        raise PolicyCorpusError(
            "Unrecognized 时效性 values; refusing to guess which laws are current: " + summary
        )

    return PolicyCorpus(
        records=active_records,
        source_count=len(rows),
        status_counts=dict(sorted(status_counts.items())),
        category_counts={
            category: category_counts.get(category, 0)
            for category in ("active", "repealed", "superseded")
        },
        unknown_statuses={},
    )


def _field(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _split_body(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    sentence_marks = "。！？；\n"
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary_floor = start + chunk_size // 2
            candidates = [normalized.rfind(mark, boundary_floor, end) for mark in sentence_marks]
            boundary = max(candidates)
            if boundary >= boundary_floor:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_policy_records(
    records: list[dict[str, Any]], *, chunk_size: int = 800, overlap: int = 120
) -> list[dict[str, Any]]:
    """Chunk each law independently and copy citation metadata onto every chunk."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    chunks: list[dict[str, Any]] = []
    for row_number, record in enumerate(records, start=1):
        body = _field(record, "政策正文")
        if not body:
            continue

        metadata = {
            "policy_id": _field(record, "政策ID") or str(row_number),
            "title": _field(record, "政策名称"),
            "publish_date": _field(record, "发布日期"),
            "document_number": _field(record, "政策文号"),
            "category": _field(record, "政策类别"),
            "issuer": _field(record, "发布单位"),
            "effect_level": _field(record, "效力级别"),
            "status": _field(record, "时效性"),
            "url": _field(record, "政策链接"),
        }
        for chunk_number, text in enumerate(_split_body(body, chunk_size, overlap), start=1):
            chunks.append(
                {
                    "text": text,
                    "metadata": {**metadata, "chunk_number": chunk_number},
                }
            )
    return chunks
