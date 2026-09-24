"""Page-aware PDF chunks and a compact vector + BM25 retriever."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATA_DIR = Path.home() / ".unravel" / "finance-rag"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = Path.home() / ".unravel" / "models" / "bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_SECTION = re.compile(r"^第[一二三四五六七八九十百\d]+节\s*(.{2,45})")
_WORD = re.compile(r"[\u4e00-\u9fff]+|[a-z]+|\d[\d,]*(?:\.\d+)?")
_COLUMNS = re.compile(r"\s{2,}")


def load_model() -> Any:
    """Load BGE from the local cache when available, or download it once."""
    from sentence_transformers import SentenceTransformer

    source = (
        str(MODEL_DIR)
        if (MODEL_DIR / "model.safetensors").is_file()
        else "BAAI/bge-small-zh-v1.5"
    )
    return SentenceTransformer(source)


def _format_line(line: str) -> tuple[str, bool]:
    cells = [cell.strip() for cell in _COLUMNS.split(line.strip()) if cell.strip()]
    if len(cells) >= 3:
        return "表格行 | " + " | ".join(cells) + " |", True
    return line.strip(), False


def extract_chunks(pdf_path: Path, report: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    """Keep each PDF page as the citation unit; preserve visual table columns."""
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        check=True,
        capture_output=True,
    )
    pages = result.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages:
        raise ValueError(f"PDF 没有可提取的文字：{pdf_path}")

    chunks: list[dict[str, Any]] = []
    section = "报告正文"
    for page_number, page in enumerate(pages, start=1):
        lines: list[tuple[str, bool]] = []
        for original in page.splitlines():
            line = original.strip()
            if not line or (line.isdigit() and len(line) <= 4):
                continue
            if "半年度报告全文" in line and report["company"] in line:
                continue
            match = _SECTION.match(line)
            if match:
                section = line[:55]
            lines.append(_format_line(original))

        part: list[str] = []
        part_is_table = False

        def flush() -> None:
            nonlocal part, part_is_table
            if not part:
                return
            body = "\n".join(part).strip()
            if len(body) >= 25:
                metadata = {
                    "company": report["company"],
                    "code": report["code"],
                    "period": report["period"],
                    "section": section,
                    "page": page_number,
                    "url": report["url"],
                    "table": part_is_table,
                }
                prefix = (
                    f"{report['company']}（{report['code']}）{report['period']}，"
                    f"{section}，PDF第{page_number}页\n"
                )
                chunks.append({"text": prefix + body, "metadata": metadata})
            part = []
            part_is_table = False

        for line, is_table in lines:
            if part and sum(map(len, part)) + len(line) > 410:
                flush()
            part.append(line)
            part_is_table = part_is_table or is_table
        flush()

    if len(chunks) < 10:
        raise ValueError(f"PDF 提取的文字过少，请检查是否为扫描件：{pdf_path}")
    return chunks, len(pages)


def tokenize(text: str) -> list[str]:
    """Use Chinese character pairs and full numbers for BM25 without extra packages."""
    terms: list[str] = []
    for match in _WORD.finditer(text.lower()):
        word = match.group()
        if "\u4e00" <= word[0] <= "\u9fff":
            terms.extend(word[i : i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                terms.append(word)
        else:
            terms.append(word)
    return terms


@dataclass
class SearchHit:
    index: int
    score: float
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None
    bm25_rank: int | None


class FinanceIndex:
    def __init__(self, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("文本块数与向量数不一致")
        self.chunks = chunks
        self.vectors = vectors
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk["text"]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)

    def bm25_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        count = len(self.chunks)
        for term in set(tokenize(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = math.log1p((count - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                scale = 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * 2.5 / (frequency + scale)
        return scores

    def search(self, query: str, model: Any, k: int = 8) -> list[SearchHit]:
        query_vector = np.asarray(
            model.encode(QUERY_PREFIX + query, normalize_embeddings=True),
            dtype=np.float32,
        )
        dense_scores = self.vectors @ query_vector
        sparse_scores = self.bm25_scores(query)
        candidate_count = min(max(k * 4, 30), len(self.chunks))
        dense_ids = np.argsort(dense_scores)[-candidate_count:][::-1].tolist()
        sparse_ids = [
            int(index)
            for index in np.argsort(sparse_scores)[-candidate_count:][::-1]
            if sparse_scores[index] > 0
        ]
        dense_ranks = {index: rank for rank, index in enumerate(dense_ids, 1)}
        sparse_ranks = {index: rank for rank, index in enumerate(sparse_ids, 1)}
        fused: dict[int, float] = defaultdict(float)
        for index, rank in dense_ranks.items():
            fused[index] += 1 / (60 + rank)
        for index, rank in sparse_ranks.items():
            fused[index] += 1 / (60 + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            SearchHit(
                index=index,
                score=fused[index],
                text=self.chunks[index]["text"],
                metadata=self.chunks[index]["metadata"],
                dense_rank=dense_ranks.get(index),
                bm25_rank=sparse_ranks.get(index),
            )
            for index in ordered
        ]


def load_index() -> FinanceIndex:
    with (INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream]
    vectors = np.load(INDEX_DIR / "vectors.npy", allow_pickle=False)
    return FinanceIndex(chunks, vectors)


def answer_question(question: str, hits: list[SearchHit], api_key: str) -> str:
    """Ask DeepSeek to cite only the supplied report pages."""
    from openai import OpenAI

    if os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"):
        for name in ("ALL_PROXY", "all_proxy"):
            if os.getenv(name, "").startswith("socks://"):
                os.environ.pop(name, None)

    context = "\n\n".join(f"[{number}] {hit.text}" for number, hit in enumerate(hits, 1))
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=90)
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是财报问答助手。只根据给定的财报片段回答。每个事实或数字后标注来源编号，如[1]。"
                    "比较数字时先核对公司、期间和单位。证据不够就明确说无法判断，不得编造。"
                ),
            },
            {"role": "user", "content": f"问题：{question}\n\n财报片段：\n{context}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def link_citations(answer: str, hits: list[SearchHit]) -> str:
    def replacement(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(hits):
            return match.group(0)
        meta = hits[number - 1].metadata
        return f"[[{number}]]({meta['url']}#page={meta['page']})"

    return re.sub(r"\[(\d+)\]", replacement, answer)
