"""DeepSeek answer generation with policy-level citations."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

from unravel.services.llm import RAGContext, get_model
from unravel.services.policy_index import PolicyHit


DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
POLICY_SYSTEM_PROMPT = """You answer questions about Shanghai policies using only
the supplied excerpts.
For each factual statement, cite the supplied source number using square brackets, such as [1].
Use only source numbers that appear in the context. If the excerpts do not answer the question,
say what is missing. Do not infer that a policy is current beyond the metadata supplied here."""


def _safe_source_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if urlparse(url).scheme in {"http", "https"} else ""


def build_cited_context(
    hits: list[PolicyHit],
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Deduplicate source laws and attach stable citation numbers to chunks."""
    source_numbers: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    context_chunks: list[str] = []
    scores: list[float] = []

    for hit in hits:
        metadata = hit.metadata
        policy_id = str(metadata.get("policy_id") or metadata.get("title") or hit.index)
        if policy_id not in source_numbers:
            source_number = len(source_numbers) + 1
            source_numbers[policy_id] = source_number
            sources.append(
                {
                    "number": source_number,
                    "policy_id": policy_id,
                    "title": metadata.get("title") or "未命名法规",
                    "publish_date": metadata.get("publish_date", ""),
                    "document_number": metadata.get("document_number", ""),
                    "effect_level": metadata.get("effect_level", ""),
                    "url": _safe_source_url(metadata.get("url")),
                }
            )

        number = source_numbers[policy_id]
        title = metadata.get("title") or "未命名法规"
        context_chunks.append(
            f"Source [{number}] — {title}; "
            f"发布日期：{metadata.get('publish_date') or '未知'}; "
            f"文号：{metadata.get('document_number') or '未知'}\n{hit.text}"
        )
        scores.append(hit.score)

    return context_chunks, sources, scores


def stream_policy_answer(
    query: str,
    hits: list[PolicyHit],
) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
    """Create a DeepSeek stream using the process environment for its key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the server environment")

    context_chunks, sources, scores = build_cited_context(hits)
    model = get_model(
        provider="OpenAI-Compatible",
        model=os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        temperature=0.2,
    )
    context = RAGContext(query=query, chunks=context_chunks, scores=scores)
    return model.stream(context, POLICY_SYSTEM_PROMPT), sources
