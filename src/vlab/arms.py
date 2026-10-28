"""The experiment arms.

Each arm answers the same questions with the same output format and decoding
settings; they differ only in where the domain knowledge comes from.

===========  =================================================================
arm          knowledge source
===========  =================================================================
prompt_only  nothing (parametric memory only)
rag          retrieved abstracts of the base corpus (base model)
sft          LoRA fine-tuning (knowledge pushed into the parameters)
sft_rag      LoRA fine-tuning + retrieval (the combination)
rag_gold     the gold abstract of each test question (oracle upper bound)
===========  =================================================================
"""

from __future__ import annotations

import time

from vlab import prompts
from vlab.utils import get_logger

LOGGER = get_logger("vlab.arms")

ARM_ORDER = ["prompt_only", "rag", "sft", "sft_rag", "rag_gold"]
RETRIEVAL_ARMS = {"rag", "sft_rag", "rag_gold"}


def _few_shot_examples(cfg, n: int) -> list[dict]:
    if n <= 0:
        return []
    from vlab.utils import read_jsonl

    train = read_jsonl(cfg.data_dir / "train.jsonl")[:n]
    return [
        {"question": r["question"], "verdict": r["verdict"], "conclusion": r["conclusion"]}
        for r in train
        if r.get("conclusion")
    ]


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[:limit]
    return head[: head.rfind(" ")] + " ..." if " " in head else head + " ..."


def build_prompts(
    cfg,
    rows: list[dict],
    arm: str,
    retriever=None,
    few_shot: list[dict] | None = None,
) -> tuple[list[str], list[list[dict]]]:
    """Render one prompt per test record, plus the evidence actually used."""
    use_retrieval = arm in RETRIEVAL_ARMS and arm != "rag_gold"
    documents_per_row: list[list[dict]] = [[] for _ in rows]

    if use_retrieval:
        questions = [r["question"] for r in rows]
        hits = retriever.batch_retrieve(questions, cfg.retriever.top_k, cfg.retriever.batch_size)
        documents_per_row = [[{"text": h["text"], "score": h["score"]} for h in row] for row in hits]
    elif arm == "rag_gold":
        documents_per_row = [[{"text": r["context"], "score": 1.0}] for r in rows]

    clip_at = cfg.retriever.max_doc_chars
    rendered: list[str] = []
    for row, docs in zip(rows, documents_per_row):
        texts = [_clip(d["text"], clip_at) for d in docs]
        if texts:
            rendered.append(prompts.rag_prompt(row["question"], texts, few_shot))
        else:
            rendered.append(prompts.prompt_only(row["question"], few_shot))
    return rendered, documents_per_row


def run_arm(
    cfg,
    arm: str,
    test_rows: list[dict],
    retriever=None,
    adapter_path: str | None = None,
    generator=None,
    few_shot: int = 0,
) -> dict:
    """Run one arm end-to-end and return predictions plus timing."""
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_ORDER}")
    if arm in RETRIEVAL_ARMS and arm != "rag_gold" and retriever is None:
        raise ValueError(f"arm {arm!r} requires a retriever")

    from vlab.backends import build_generator

    if generator is None:
        generator = build_generator(cfg, adapter_path=adapter_path)

    few_shot_examples = _few_shot_examples(cfg, few_shot)
    rendered, documents_per_row = build_prompts(cfg, test_rows, arm, retriever, few_shot_examples)

    LOGGER.info("arm=%s | %s prompts | adapter=%s", arm, len(rendered), adapter_path or "none")
    start = time.perf_counter()
    raw_outputs = generator.generate(rendered)
    elapsed = time.perf_counter() - start

    predictions = []
    for row, raw, docs in zip(test_rows, raw_outputs, documents_per_row):
        verdict, conclusion = prompts.parse_prediction(raw)
        predictions.append(
            {
                "pid": row.get("pid", ""),
                "arm": arm,
                "question": row["question"],
                "gold_verdict": row["verdict"],
                "pred_verdict": verdict,
                "gold_conclusion": row["conclusion"],
                "pred_conclusion": conclusion,
                "correct": int(verdict == row["verdict"]),
                "raw": raw,
                "documents": docs,
            }
        )
    LOGGER.info("arm=%s finished in %.1fs", arm, elapsed)
    return {
        "arm": arm,
        "seconds": round(elapsed, 2),
        "n": len(predictions),
        "predictions": predictions,
    }
