"""Dense retrieval over the domain corpus (bi-encoder + FAISS).

The retriever is deliberately simple: encode documents once with a small
bi-encoder, store them in a flat inner-product index, and return the top-k
abstracts for a question. A flat index keeps the demo reproducible on a laptop
(50k x 384 floats is ~75 MB) while still being exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from vlab.utils import get_logger, read_jsonl

LOGGER = get_logger("vlab.retriever")


def _document_text(row: dict) -> str:
    question = row.get("question", "").strip()
    context = row.get("context", "").strip()
    if question and context:
        return f"{question}\n{context}"
    return question or context


class Retriever:
    """FAISS-backed dense retriever over the abstract corpus."""

    def __init__(self, documents: list[dict], index, model_name: str) -> None:
        self.documents = documents
        self.index = index
        self.model_name = model_name
        self._encoder = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, corpus_rows: list[dict], cfg, show_progress: bool = True) -> "Retriever":
        from sentence_transformers import SentenceTransformer

        texts = [_document_text(r) for r in corpus_rows]
        LOGGER.info("encoding %s documents with %s", len(texts), cfg.retriever.embedding_model)
        encoder = SentenceTransformer(cfg.retriever.embedding_model, device=cfg.device)
        embeddings = encoder.encode(
            texts,
            batch_size=cfg.retriever.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=cfg.retriever.normalize,
            show_progress_bar=show_progress,
        ).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        retriever = cls(corpus_rows, index, cfg.retriever.embedding_model)
        retriever._encoder = encoder
        LOGGER.info("index built: %s vectors, dim=%s", index.ntotal, dim)
        return retriever

    # -------------------------------------------------------------------- io
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        meta = {"embedding_model": self.model_name, "documents": self.documents}
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("saved index to %s", directory)

    @classmethod
    def load(cls, directory: str | Path, model_name: str | None = None, device: str = "cpu") -> "Retriever":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "faiss.index"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        retriever = cls(meta["documents"], index, model_name or meta.get("embedding_model", ""))
        try:
            from sentence_transformers import SentenceTransformer

            retriever._encoder = SentenceTransformer(retriever.model_name, device=device)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not load encoder (%s)", exc)
        return retriever

    # ---------------------------------------------------------------- search
    def _encode_queries(self, queries: list[str], batch_size: int) -> np.ndarray:
        return self._encoder.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def batch_retrieve(self, questions: list[str], top_k: int, batch_size: int = 64) -> list[list[dict]]:
        if self._encoder is None:
            raise RuntimeError("retriever encoder is not initialised")
        vectors = self._encode_queries(questions, batch_size)
        scores, ids = self.index.search(vectors, top_k)
        results: list[list[dict]] = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for score, doc_id in zip(row_scores, row_ids):
                if doc_id < 0:
                    continue
                doc = self.documents[int(doc_id)]
                hits.append(
                    {
                        "pid": doc.get("pid", ""),
                        "text": _document_text(doc),
                        "score": float(score),
                    }
                )
            results.append(hits)
        return results

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        return self.batch_retrieve([question], top_k)[0]


def build_and_save(cfg) -> Retriever:
    """Build the retrieval index from ``data/corpus.jsonl`` and persist it."""
    from vlab.config import setup_hf_endpoint

    setup_hf_endpoint(cfg.hf_endpoint)
    cfg.ensure_dirs()
    corpus = read_jsonl(cfg.data_dir / "corpus.jsonl")
    if not corpus:
        raise FileNotFoundError(
            f"corpus not found at {cfg.data_dir / 'corpus.jsonl'}; run scripts/prepare_data.py first"
        )
    retriever = Retriever.build(corpus, cfg)
    retriever.save(cfg.data_dir / "index")
    return retriever
