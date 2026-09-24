"""Classroom walkthrough for the Shanghai policy RAG index."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st

from unravel.services.embedders import get_embedder
from unravel.services.policy_index import (
    INDEX_DIRECTORY,
    MODEL_DIRECTORY,
    PPT_EXPECTED_CHUNK_COUNT,
    PolicyHit,
    PolicyIndex,
    QUERY_INSTRUCTION,
    load_policy_index,
)
from unravel.services.policy_rag import stream_policy_answer
from unravel.utils.ui import render_page_header, render_section_heading
from unravel.utils.visualization import create_embedding_plot


@st.cache_resource(show_spinner=False)
def _load_cached_index(index_path: str) -> PolicyIndex:
    return load_policy_index(index_path)


def _map_frame(index: PolicyIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index.coordinates, columns=["x", "y"])
    frame["chunk_index"] = range(len(index.chunks))
    frame["cluster_label"] = index.clusters
    previews: list[str] = []
    titles: list[str] = []
    categories: list[str] = []
    for chunk in index.chunks:
        metadata = chunk["metadata"]
        title = str(metadata.get("title") or "未命名法规")
        titles.append(title)
        categories.append(str(metadata.get("category") or "未分类"))
        preview = str(chunk["text"]).replace("\n", " ")[:150]
        previews.append(f"《{title}》 · {preview}")
    frame["policy_title"] = titles
    frame["category"] = categories
    frame["text_preview"] = previews
    return frame


def _selected_chunk_index(event: Any) -> int | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection", {})
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points", [])

    for point in points or []:
        curve_number = point.get("curve_number", point.get("curveNumber", 0))
        if curve_number == 0:
            return int(point.get("point_index", point.get("pointIndex", -1)))
        if curve_number == 1:
            customdata = point.get("customdata")
            if isinstance(customdata, (list, tuple)):
                customdata = customdata[0]
            if customdata is not None:
                return int(customdata)
    return None


def _draw_map(
    placeholder: Any,
    frame: pd.DataFrame,
    *,
    query_point: dict[str, float] | None,
    neighbors: list[int],
    key: str,
) -> int | None:
    figure = create_embedding_plot(
        frame,
        x_col="x",
        y_col="y",
        color_col="cluster_label",
        hover_data=["policy_title", "category", "chunk_index"],
        query_point=query_point,
        neighbors_indices=neighbors,
        title="",
    )
    with placeholder.container():
        st.markdown("#### 法规文本块语义地图")
        st.caption(
            "每个点代表一个文本块。点击点可查看法规信息和该块原文。"
        )
        event = st.plotly_chart(
            figure,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=key,
        )
    return _selected_chunk_index(event)


def _hits_from_state(index: PolicyIndex, raw_hits: list[dict[str, Any]]) -> list[PolicyHit]:
    return [
        PolicyHit(
            index=int(item["index"]),
            score=float(item["score"]),
            text=str(index.chunks[int(item["index"])]["text"]),
            metadata=dict(index.chunks[int(item["index"])]["metadata"]),
        )
        for item in raw_hits
    ]


def _render_retrieved_hits(hits: list[PolicyHit]) -> None:
    render_section_heading("召回文本块", "按余弦相似度从高到低排列。")
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.metadata
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {metadata.get('title') or '未命名法规'}**  "
                f"· 余弦相似度 `{hit.score:.4f}`"
            )
            st.caption(
                " · ".join(
                    value
                    for value in (
                        str(metadata.get("effect_level") or ""),
                        str(metadata.get("publish_date") or ""),
                        str(metadata.get("document_number") or ""),
                    )
                    if value
                )
            )
            st.write(hit.text)


def _render_selected_chunk(index: PolicyIndex, selected_index: int | None) -> None:
    if selected_index is None or not 0 <= selected_index < len(index.chunks):
        return
    chunk = index.chunks[selected_index]
    metadata = chunk["metadata"]
    render_section_heading("所选文本块原文")
    with st.container(border=True):
        st.markdown(f"**{metadata.get('title') or '未命名法规'}**")
        st.caption(
            " · ".join(
                value
                for value in (
                    str(metadata.get("effect_level") or ""),
                    str(metadata.get("publish_date") or ""),
                    str(metadata.get("document_number") or ""),
                    f"第 {metadata.get('chunk_number', '?')} 块",
                )
                if value
            )
        )
        st.write(chunk["text"])
        url = str(metadata.get("url") or "")
        if url.startswith(("https://", "http://")):
            st.link_button("打开法规原文", url)


def _citation_markdown(answer: str, sources: list[dict[str, Any]]) -> str:
    by_number = {int(source["number"]): source for source in sources}

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        source = by_number.get(number)
        if source and source.get("url"):
            url = quote(str(source["url"]), safe=":/?&=#%")
            return f"[[{number}]]({url})"
        return match.group(0)

    return re.sub(r"\[(\d+)\]", replace, answer)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    render_section_heading("引用法规")
    for source in sources:
        label = f"[{source['number']}] {source['title']}"
        details = " · ".join(
            str(value)
            for value in (
                source.get("effect_level"),
                source.get("publish_date"),
                source.get("document_number"),
            )
            if value
        )
        if source.get("url"):
            st.link_button(label, source["url"], use_container_width=True)
        else:
            st.markdown(f"**{label}**")
        if details:
            st.caption(details)


def _run_query(
    question: str,
    index: PolicyIndex,
    frame: pd.DataFrame,
    map_placeholder: Any,
    timeline_placeholder: Any,
    vector_placeholder: Any,
) -> tuple[list[PolicyHit], str, list[dict[str, Any]], int | None, dict[str, float]]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set in the server environment")

    timeline_placeholder.info("1 / 4 · 正在把问题转换为向量")
    query_vector = get_embedder(str(MODEL_DIRECTORY)).embed_query(
        QUERY_INSTRUCTION + question
    )
    dimension = int(query_vector.shape[0])
    prefix = ", ".join(f"{number:.6f}" for number in query_vector[:8])
    vector_placeholder.code(
        f"问题向量维度：{dimension}\n前 8 个值：[{prefix}]",
        language="text",
    )

    x_coord, y_coord = index.project(query_vector)
    query_point = {"x": x_coord, "y": y_coord}
    run_id = int(st.session_state.get("policy_query_run_id", 0)) + 1
    st.session_state.policy_query_run_id = run_id
    selected_index = _draw_map(
        map_placeholder,
        frame,
        query_point=query_point,
        neighbors=[],
        key=f"policy-map-{run_id}-query",
    )
    time.sleep(0.7)

    timeline_placeholder.info("2 / 4 · 在语义地图上定位问题并计算余弦相似度")
    hits = index.search(query_vector, k=6)
    selected_index = _draw_map(
        map_placeholder,
        frame,
        query_point=query_point,
        neighbors=[hit.index for hit in hits],
        key=f"policy-map-{run_id}",
    )
    time.sleep(0.7)

    timeline_placeholder.info("3 / 4 · 已召回周围 6 个文本块")
    _render_retrieved_hits(hits)
    time.sleep(0.7)

    timeline_placeholder.info("4 / 4 · 正在把召回内容交给 DeepSeek")
    stream, sources = stream_policy_answer(question, hits)
    answer_placeholder = st.empty()
    with answer_placeholder.container():
        answer_value = st.write_stream(stream)
    answer = answer_value if isinstance(answer_value, str) else "".join(answer_value)
    timeline_placeholder.success("回答已生成")
    answer_placeholder.markdown(_citation_markdown(answer, sources))
    _render_sources(sources)
    return hits, answer, sources, selected_index, query_point


def render_policy_step() -> None:
    """Render the separate policy classroom demo without changing other RAG steps."""
    index_path = Path(os.getenv("POLICY_INDEX_DIR", str(INDEX_DIRECTORY))).expanduser()
    render_page_header(
        "上海政策法规 RAG",
        "查看法规文本块的语义分布，并按步骤观察提问、召回和回答。",
        "Classroom walkthrough",
    )

    try:
        index = _load_cached_index(str(index_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.warning(f"法规索引尚未就绪：{exc}")
        st.code(
            "python -m unravel.policy_download_cli\n"
            "python -m unravel.policy_cli",
            language="bash",
        )
        st.caption(
            "先从上海市官方法规库下载公开数据，再构建索引；"
            "索引生成后，服务启动时只读取本地文件。"
        )
        return

    manifest = index.manifest
    metrics = st.columns(4)
    metrics[0].metric("当前法规", f"{manifest['active_law_count']:,}")
    metrics[1].metric("文本块", f"{manifest['chunk_count']:,}")
    metrics[2].metric("向量维度", f"{manifest['embedding_dimension']:,}")
    metrics[3].metric("建索引用时", f"{manifest['indexing_seconds']:.2f}s")

    counts = manifest.get("category_counts", {})
    st.caption(
        f"原始法规 {manifest['source_law_count']:,} 部；"
        f"失效废止 {counts.get('repealed', 0):,} 部；"
        f"修改后重新发布的旧版本 {counts.get('superseded', 0):,} 部。"
    )
    if (
        manifest.get("ppt_status_count_check") != "match"
        or manifest.get("ppt_chunk_count_check") != "match"
    ):
        st.warning(
            "本地语料或切块结果与 PPT 记录不完全一致；PPT 参考值为 "
            f"1,345 部现行法规、{PPT_EXPECTED_CHUNK_COUNT:,} 个文本块。"
        )

    with st.form("policy_query_form", clear_on_submit=False):
        question = st.text_input(
            "提出问题",
            placeholder="例如：哪些规定涉及企业开办登记？",
            label_visibility="collapsed",
        )
        ask = st.form_submit_button("提问", type="primary", width="stretch")

    timeline_placeholder = st.empty()
    vector_placeholder = st.empty()
    map_placeholder = st.empty()
    frame = _map_frame(index)
    selected_index: int | None = None
    hits: list[PolicyHit] = []
    answer = ""
    sources: list[dict[str, Any]] = []

    if ask:
        if not question.strip():
            st.warning("先输入一个问题。")
        else:
            try:
                hits, answer, sources, selected_index, query_point = _run_query(
                    question.strip(),
                    index,
                    frame,
                    map_placeholder,
                    timeline_placeholder,
                    vector_placeholder,
                )
                st.session_state.policy_query_state = {
                    "question": question.strip(),
                    "query_point": [query_point["x"], query_point["y"]],
                    "hits": [{"index": hit.index, "score": hit.score} for hit in hits],
                    "answer": answer,
                    "sources": sources,
                }
            except Exception as exc:
                st.error(f"查询失败：{exc}")
    else:
        state = st.session_state.get("policy_query_state", {})
        query_point_data = state.get("query_point")
        query_point = (
            {"x": float(query_point_data[0]), "y": float(query_point_data[1])}
            if query_point_data
            else None
        )
        hits = _hits_from_state(index, state.get("hits", []))
        selected_index = _draw_map(
            map_placeholder,
            frame,
            query_point=query_point,
            neighbors=[hit.index for hit in hits],
            key=f"policy-map-{int(st.session_state.get('policy_query_run_id', 0))}",
        )
        if selected_index is None:
            selected_index = st.session_state.get("policy_selected_chunk_index")
        elif selected_index >= 0:
            st.session_state.policy_selected_chunk_index = selected_index

        answer = str(state.get("answer", ""))
        sources = list(state.get("sources", []))
        if hits:
            _render_retrieved_hits(hits)
        if answer:
            render_section_heading("回答")
            st.markdown(_citation_markdown(answer, sources))
            _render_sources(sources)

    _render_selected_chunk(index, selected_index)
