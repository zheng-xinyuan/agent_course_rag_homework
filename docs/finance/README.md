# 财报问答知识库：实现与复现

[`reports.json`](../../unravel/finance/reports.json) 列出从巨潮资讯取得的 10 家食品公司 **2025 年半年度报告全文**，每家公司一份。数据来自原始 PDF，而非报告摘要。实现代码位于 [`unravel/finance`](../../unravel/finance/)。

## 实现

- `build.py` 下载 PDF，用系统 `pdftotext -layout` 按 PDF 页提取文字。表格的行保留为独立的 `表格行 | 列1 | 列2 | …` 文本；每个块记录公司、证券代码、期间、章节、PDF 页码和官方 PDF 链接。
- 使用本地 `BAAI/bge-small-zh-v1.5` 生成向量；BM25 用中文双字词和数字检索；两种排名用 RRF 合并。
- `app.py` 展示问答页面、检索到的原文块及其 PDF 页码链接。DeepSeek 回答中的 `[1]` 等编号会变成对应原文页的链接。
- `evaluate.py` 跑 `questions.json` 中的 10 道题，其中第 9、10 题跨公司，记录答案和召回块到 `evaluation.json`。

实测与交付材料：

- [10 题召回、答案和人工判定](../../unravel/finance/evaluation.json)
- [问答页面截图](../../deliverables/qa.png)
- [一页结论 PDF](../../deliverables/conclusion.pdf)

## 在当前机器运行

在仓库根目录执行：

```bash
env -u PYTHONPATH ~/.unravel/policy-venv/bin/python -m unravel.finance.build
env -u PYTHONPATH ~/.unravel/policy-venv/bin/python -m unravel.cli --port 8503
```

已有索引时只需第二条命令。当前机器复用已有 conda/PyTorch、向量模型和隔离环境。其他机器需要 `pdftotext`（Poppler）及 Python 包 `numpy`、`requests`、`sentence-transformers`、`streamlit`、`python-dotenv`、`openai`、`click`；没有本地模型时会下载 `BAAI/bge-small-zh-v1.5`。

DeepSeek 密钥只放在仓库根目录的 `.env`：

```text
DEEPSEEK_API_KEY=你的密钥
```

`.env` 已被仓库的 `.gitignore` 忽略。通过 `unravel` 入口运行时，也可以从当前机器已有的 OpenCode 凭证读取 DeepSeek 密钥。

下载的 PDF 和索引位于 `~/.unravel/finance-rag/`，不会进入代码仓库。10 份 PDF 约 24 MB，索引约 16 MB。

## 已知边界

PDF 按页面保留来源，表格按版面空白分隔列；合并单元格和跨页表格可能需要人工核对。`evaluation.json` 保留实际召回结果、模型回答和人工判定：8 道单公司题正确，第 9 题错误，第 10 题部分正确。跨公司题的主要问题是召回块集中在少数公司。正确性与错误原因以 PDF 原文为准。
