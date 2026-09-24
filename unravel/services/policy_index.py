"""Build and load the one-time vector and map index for policy RAG."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import umap
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from unravel.services.embedders import Embedder
from unravel.services.policy_corpus import (
    PPT_EXPECTED_COUNTS,
    chunk_policy_records,
    load_policy_corpus,
)


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
MODEL_DIRECTORY = Path.home() / ".unravel" / "models" / "bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
INDEX_DIRECTORY = Path.home() / ".unravel" / "policy-index"
INDEX_FORMAT_VERSION = 3
PPT_EXPECTED_CHUNK_COUNT = 9353
MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def ensure_policy_model() -> Path:
    """Download only the files used by the local sentence-transformers model."""
    if all((MODEL_DIRECTORY / name).is_file() for name in MODEL_FILES):
        return MODEL_DIRECTORY

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=EMBEDDING_MODEL,
        local_dir=MODEL_DIRECTORY,
        allow_patterns=list(MODEL_FILES),
    )
    return MODEL_DIRECTORY


@dataclass
class PolicyHit:
    """A retrieved source chunk and its cosine similarity."""

    index: int
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass
class PolicyIndex:
    """In-memory policy vectors, chunks, and their fitted map projection."""

    chunks: list[dict[str, Any]]
    vectors: NDArray[np.float32]
    coordinates: NDArray[np.float32]
    clusters: NDArray[np.int32]
    reducer: Any
    manifest: dict[str, Any]

    def search(self, query_vector: NDArray[Any], k: int = 6) -> list[PolicyHit]:
        if self.vectors.size == 0 or k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Query vector has {query.shape[0]} dimensions; expected {self.vectors.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise ValueError("Query embedding has zero length")
        scores = self.vectors @ (query / norm)
        count = min(k, len(scores))
        candidate_ids = np.argpartition(scores, -count)[-count:]
        ordered_ids = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
        return [
            PolicyHit(
                index=int(index),
                score=float(scores[index]),
                text=str(self.chunks[index]["text"]),
                metadata=dict(self.chunks[index]["metadata"]),
            )
            for index in ordered_ids
        ]

    def project(self, query_vector: NDArray[Any]) -> tuple[float, float]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        projected = self.reducer.transform(query)[0]
        return float(projected[0]), float(projected[1])


def _write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_policy_index(
    source_path: str | Path,
    output_dir: str | Path = INDEX_DIRECTORY,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Filter the corpus, embed chunks, fit UMAP, and persist the index."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    started = time.monotonic()
    corpus = load_policy_corpus(source)
    chunks = chunk_policy_records(corpus.records, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No non-empty policy text was found in the current-law records")

    expected_mismatches = {
        key: {"expected": expected, "actual": corpus.category_counts.get(key, 0)}
        for key, expected in PPT_EXPECTED_COUNTS.items()
        if corpus.category_counts.get(key, 0) != expected
    }
    if expected_mismatches:
        print(
            "PPT status-count check differs: "
            f"{json.dumps(expected_mismatches, ensure_ascii=False)}"
        )

    embedder = Embedder(str(ensure_policy_model()))
    vectors = np.asarray(
        embedder.embed_texts(
            [chunk["text"] for chunk in chunks],
            batch_size=batch_size,
            show_progress=True,
            normalize=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding response count does not match the number of text chunks")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(chunks) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        transform_seed=42,
    )
    coordinates = np.asarray(reducer.fit_transform(vectors), dtype=np.float32)
    cluster_count = min(len(chunks), 10, max(2, int(np.sqrt(len(chunks) / 2))))
    clusters = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(
        coordinates
    ).astype(np.int32)
    chunk_count_matches_ppt = len(chunks) == PPT_EXPECTED_CHUNK_COUNT

    manifest: dict[str, Any] = {
        "format_version": INDEX_FORMAT_VERSION,
        "source_name": source.name,
        "source_sha256": _sha256_file(source),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "source_law_count": corpus.source_count,
        "active_law_count": len(corpus.records),
        "status_counts": corpus.status_counts,
        "category_counts": corpus.category_counts,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(vectors.shape[1]),
        "map_projection": "UMAP-2D",
        "map_cluster_count": cluster_count,
        "indexing_seconds": 0,
        "ppt_status_count_check": "match" if not expected_mismatches else "mismatch",
        "ppt_chunk_count_check": "match" if chunk_count_matches_ppt else "mismatch",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        np.save(staging / "vectors.npy", vectors)
        np.save(staging / "coordinates.npy", coordinates)
        np.save(staging / "clusters.npy", clusters)
        _write_chunks(staging / "chunks.jsonl", chunks)
        joblib.dump(reducer, staging / "umap.joblib")
        manifest["indexing_seconds"] = round(time.monotonic() - started, 2)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except OSError:
            if backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    manifest["indexing_seconds"] = round(time.monotonic() - started, 2)
    manifest_tmp = output / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, output / "manifest.json")

    return manifest


def load_policy_index(index_dir: str | Path = INDEX_DIRECTORY) -> PolicyIndex:
    """Load a previously built index without contacting any model API."""
    directory = Path(index_dir).expanduser()
    required = [
        "manifest.json",
        "chunks.jsonl",
        "vectors.npy",
        "coordinates.npy",
        "clusters.npy",
        "umap.joblib",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Policy index is incomplete at {directory}: {', '.join(missing)}")

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError("Policy index format is not supported by this version")
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Policy index was built with a different embedding model")
    chunks = _read_chunks(directory / "chunks.jsonl")
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    coordinates = np.load(directory / "coordinates.npy", allow_pickle=False)
    clusters = np.load(directory / "clusters.npy", allow_pickle=False)
    reducer = joblib.load(directory / "umap.joblib")

    if (
        vectors.shape[0] != len(chunks)
        or coordinates.shape != (len(chunks), 2)
        or clusters.shape != (len(chunks),)
    ):
        raise ValueError("Policy index files have inconsistent row counts")
    return PolicyIndex(chunks, vectors, coordinates, clusters, reducer, manifest)
