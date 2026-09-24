"""Run ten report questions and record answers plus retrieved chunks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from unravel.finance.core import answer_question, load_index, load_model


HERE = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("请先在根目录 .env 配置 DEEPSEEK_API_KEY")

    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    index = load_index()
    model = load_model()
    results = []
    for item in questions:
        hits = index.search(item["question"], model, k=10)
        try:
            answer = answer_question(item["question"], hits, key)
            error = ""
        except Exception as exc:
            answer = ""
            error = str(exc)
        results.append(
            {
                **item,
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": hit.index,
                        "company": hit.metadata["company"],
                        "section": hit.metadata["section"],
                        "page": hit.metadata["page"],
                        "url": hit.metadata["url"] + f"#page={hit.metadata['page']}",
                    }
                    for rank, hit in enumerate(hits, 1)
                ],
                "answer": answer,
                "correct": None,
                "issue": error,
            }
        )
        (HERE / "evaluation.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"第 {item['id']} 题完成：{answer[:90] or error}", flush=True)


if __name__ == "__main__":
    main()
