"""PubMedQA loading and normalization.

PubMedQA is a biomedical QA dataset built from PubMed abstracts. Every record
reframes a paper title into a research question ("Does X predict Y?") and pairs
it with the abstract sections and a yes/no/maybe verdict.

Splits used here:

* ``pqa_labeled``    -- 1k expert-annotated records, used as the held-out test set.
* ``pqa_artificial`` -- 211k automatically labelled records, used for SFT training
  and as the retrieval corpus.

The two splits are disjoint by construction, and the loader enforces that the
corpus never contains a document from the test split (see ``dedup``).
"""

from __future__ import annotations

from typing import Any

from vlab.utils import get_logger, write_jsonl

LOGGER = get_logger("vlab.data")

CONTEXT_MAX_CHARS = 1600


def _flatten_context(context: Any) -> str:
    """PubMedQA stores the abstract as a dict of section lists."""
    if isinstance(context, dict):
        parts = context.get("contexts") or []
    elif isinstance(context, list):
        parts = context
    else:
        parts = [str(context)]
    text = " ".join(str(p).strip() for p in parts if str(p).strip())
    return text[:CONTEXT_MAX_CHARS]


def normalize(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw PubMedQA record onto the internal schema."""
    question = (row.get("question") or "").strip()
    verdict = (row.get("final_decision") or "").strip().lower()
    if not question or verdict not in ("yes", "no", "maybe"):
        return None
    conclusion = (row.get("long_answer") or "").strip()
    return {
        "pid": str(row.get("pubid", "")),
        "question": question,
        "verdict": verdict,
        "conclusion": conclusion,
        "context": _flatten_context(row.get("context")),
    }


def load_split(
    dataset_name: str,
    config_name: str,
    limit: int | None = None,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Load and normalize a PubMedQA config.

    Streaming is preferred so that a subset can be pulled without downloading the
    whole 211k-row artificial split; it falls back to a full download.
    """
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    if streaming:
        try:
            stream = load_dataset(dataset_name, config_name, split="train", streaming=True)
            for raw in stream:
                item = normalize(raw)
                if item:
                    rows.append(item)
                if limit and len(rows) >= limit:
                    break
            LOGGER.info("loaded %s rows from %s (streaming)", len(rows), config_name)
            return rows
        except Exception as exc:  # noqa: BLE001 - fall back to the non-streaming path
            LOGGER.warning("streaming load failed (%s); downloading instead", exc)
            rows = []
    dataset = load_dataset(dataset_name, config_name, split="train")
    for raw in dataset:
        item = normalize(raw)
        if item:
            rows.append(item)
        if limit and len(rows) >= limit:
            break
    LOGGER.info("loaded %s rows from %s", len(rows), config_name)
    return rows


def dedup_corpus(
    corpus: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop corpus documents that would leak a test answer.

    A document is removed when it shares a pubmed id or an (almost) identical
    question with any test record. Without this filter, retrieval could surface
    the very abstract the gold answer was derived from, which inflates the RAG
    arm and makes the comparison meaningless.
    """
    banned_pids = {r["pid"] for r in test_rows if r["pid"]}
    banned_questions = {r["question"].strip().lower() for r in test_rows}
    kept, removed = [], 0
    for doc in corpus:
        if doc["pid"] and doc["pid"] in banned_pids:
            removed += 1
            continue
        if doc["question"].strip().lower() in banned_questions:
            removed += 1
            continue
        kept.append(doc)
    LOGGER.info("corpus: kept %s documents, removed %s (leakage guard)", len(kept), removed)
    return kept


def prepare(cfg) -> dict[str, str]:
    """Materialize train/test/corpus jsonl files. Returns their paths."""
    from vlab.config import setup_hf_endpoint

    setup_hf_endpoint(cfg.hf_endpoint)
    cfg.ensure_dirs()

    test_rows = load_split(cfg.data.dataset_name, cfg.data.eval_source, cfg.data.max_eval_samples)
    if cfg.data.max_eval_samples and len(test_rows) > cfg.data.max_eval_samples:
        test_rows = test_rows[: cfg.data.max_eval_samples]

    # Pull a bit more than needed so corpus and train can be disjoint-ish.
    needed = cfg.data.max_train_samples + cfg.data.corpus_size
    pool = load_split(cfg.data.dataset_name, cfg.data.train_source, needed)

    train_rows = pool[: cfg.data.max_train_samples]
    corpus_rows = pool[cfg.data.max_train_samples : cfg.data.max_train_samples + cfg.data.corpus_size]
    corpus_rows = dedup_corpus(corpus_rows, test_rows)

    paths = {
        "test": str(cfg.data_dir / "test.jsonl"),
        "train": str(cfg.data_dir / "train.jsonl"),
        "corpus": str(cfg.data_dir / "corpus.jsonl"),
    }
    write_jsonl(paths["test"], test_rows)
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["corpus"], corpus_rows)
    LOGGER.info("wrote %s / %s / %s", len(test_rows), len(train_rows), len(corpus_rows))
    return paths
