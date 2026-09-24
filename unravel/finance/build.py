"""Download ten public half-year reports and build the local search index."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from unravel.finance.core import DATA_DIR, INDEX_DIR, extract_chunks, load_model


REPORTS = Path(__file__).with_name("reports.json")


def download_report(report: dict[str, str]) -> Path:
    pdf_dir = DATA_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"{report['code']}_2025H1.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path

    temporary = path.with_suffix(".pdf.part")
    try:
        with requests.get(report["url"], timeout=45, stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
                    if stream.tell() > 20 * 1024 * 1024:
                        raise ValueError(f"PDF 大于 20 MB：{report['company']}")
        with temporary.open("rb") as stream:
            if stream.read(4) != b"%PDF":
                raise ValueError(f"下载结果不是 PDF：{report['company']}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    reports = json.loads(REPORTS.read_text(encoding="utf-8"))
    if len({report["code"] for report in reports}) < 10:
        raise ValueError("至少需要 10 家不同的公司")

    chunks = []
    page_counts = {}
    for report in reports:
        pdf = download_report(report)
        report_chunks, pages = extract_chunks(pdf, report)
        chunks.extend(report_chunks)
        page_counts[report["code"]] = pages
        print(f"{report['company']}：{pages} 页，{len(report_chunks)} 块", flush=True)

    print(f"共 {len(chunks)} 块，开始生成向量", flush=True)
    model = load_model()
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "report_count": len(reports),
                "page_counts": page_counts,
                "chunk_count": len(chunks),
                "embedding_dimension": int(vectors.shape[1]),
                "model": "BAAI/bge-small-zh-v1.5",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"索引完成：{INDEX_DIR}", flush=True)


if __name__ == "__main__":
    main()
