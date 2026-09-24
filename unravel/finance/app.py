"""Page-cited financial report question-answering interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from unravel.finance.core import (
    INDEX_DIR,
    answer_question,
    link_citations,
    load_index,
    load_model,
)


load_dotenv(Path.cwd() / ".env")
st.set_page_config(page_title="财报问答知识库", layout="wide")


@st.cache_resource(show_spinner="加载财报索引中…")
def cached_index():
    return load_index()


@st.cache_resource(show_spinner="加载本地向量模型中…")
def cached_model():
    return load_model()


st.title("财报问答知识库")
st.caption("10 家食品公司 · 2025 年半年度报告 · 向量 + BM25 · 来源定位到 PDF 页")

manifest_path = INDEX_DIR / "manifest.json"
if not manifest_path.exists():
    st.error("还没有索引。先运行：python -m unravel.finance.build")
    st.stop()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
first, second, third = st.columns(3)
first.metric("公司 / 报告", manifest["report_count"])
second.metric("PDF 页数", sum(manifest["page_counts"].values()))
third.metric("文本块", manifest["chunk_count"])

with st.form("ask"):
    question = st.text_input(
        "提问",
        placeholder="例如：涪陵榨菜和洽洽食品 2025 年上半年的营业收入分别是多少？",
    )
    ask = st.form_submit_button("提问", type="primary")

if ask and question.strip():
    index = cached_index()
    model = cached_model()
    with st.spinner("向量 + BM25 检索中…"):
        hits = index.search(question.strip(), model, k=10)

    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        with st.spinner("生成带出处的回答中…"):
            try:
                answer = answer_question(question.strip(), hits, key)
                st.subheader("回答")
                st.markdown(link_citations(answer, hits))
            except Exception as error:
                st.error(f"DeepSeek 请求失败：{error}")
    else:
        st.warning("请在仓库根目录的 .env 中填写 DEEPSEEK_API_KEY 后重启页面。")

    st.subheader("检索到的原文")
    for number, hit in enumerate(hits, 1):
        meta = hit.metadata
        url = f"{meta['url']}#page={meta['page']}"
        label = f"[{number}] {meta['company']} · {meta['section']} · PDF 第 {meta['page']} 页"
        with st.expander(label, expanded=number <= 3):
            st.markdown(f"[打开官方财报原文]({url})")
            st.caption(f"向量排名：{hit.dense_rank or '—'}；BM25 排名：{hit.bm25_rank or '—'}")
            st.text(hit.text)
