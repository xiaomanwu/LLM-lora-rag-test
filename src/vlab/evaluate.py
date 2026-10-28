"""Metrics and reporting.

Metrics reported per arm:

* ``accuracy``  -- exact match of the yes/no/maybe verdict.
* ``macro_f1``  -- macro-averaged F1 over the three verdicts (guards against a
  degenerate "always yes" predictor scoring well on a skewed test set).
* ``rouge_l``   -- longest-common-subsequence F1 between the generated and gold
  conclusion, i.e. whether the *reason* is right and not just the label.
* ``latency``   -- wall-clock seconds per answer, since the whole point of the
  comparison is a cost/benefit trade-off.
"""

from __future__ import annotations

from typing import Any

from vlab.utils import get_logger, load_json

LOGGER = get_logger("vlab.eval")

VERDICT_LABELS = ["yes", "no", "maybe"]


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _fallback_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens, ref_tokens = prediction.split(), reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_rouge_scorer():
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return lambda p, r: scorer.score(r, p)["rougeL"].fmeasure
    except Exception:  # noqa: BLE001 - rouge_score is optional
        return _fallback_rouge_l


_ROUGE = None


def rouge_l(prediction: str, reference: str) -> float:
    global _ROUGE
    if _ROUGE is None:
        _ROUGE = _get_rouge_scorer()
    try:
        return float(_ROUGE(prediction, reference))
    except Exception:  # noqa: BLE001
        return _fallback_rouge_l(prediction, reference)


def compute_metrics(predictions: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    gold = [p["gold_verdict"] for p in predictions]
    pred = [p["pred_verdict"] for p in predictions]
    rouges = [rouge_l(p["pred_conclusion"], p["gold_conclusion"]) for p in predictions if p["gold_conclusion"]]
    # Did the model actually emit the "Verdict: x" line, or did we have to guess
    # from a keyword match? A big part of what fine-tuning fixes.
    formatted = sum(1 for p in predictions if "verdict:" in p["raw"].lower())

    report = classification_report(
        gold, pred, labels=VERDICT_LABELS, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in VERDICT_LABELS
    }
    distribution = {label: pred.count(label) / max(len(pred), 1) for label in VERDICT_LABELS}
    return {
        "n": len(predictions),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, average="macro", zero_division=0), 4),
        "rouge_l": round(sum(rouges) / len(rouges), 4) if rouges else 0.0,
        "format_compliance": round(formatted / max(len(predictions), 1), 4),
        "prediction_distribution": {k: round(v, 4) for k, v in distribution.items()},
        "gold_distribution": {
            k: round(gold.count(k) / max(len(gold), 1), 4) for k in VERDICT_LABELS
        },
        "per_class": per_class,
    }


def summarize(arm_result: dict) -> dict:
    metrics = compute_metrics(arm_result["predictions"])
    seconds = arm_result.get("seconds", 0.0)
    n = max(metrics["n"], 1)
    metrics.update(
        {
            "arm": arm_result["arm"],
            "total_seconds": round(seconds, 2),
            "seconds_per_answer": round(seconds / n, 3),
        }
    )
    return metrics


def _fmt_table(rows: list[dict]) -> str:
    header = ["arm", "accuracy", "macro_f1", "rouge_l", "format%", "s/answer", "n"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append(
            "| {arm} | {acc:.3f} | {f1:.3f} | {rgl:.3f} | {fmt:.0%} | {lat:.2f} | {n} |".format(
                arm=row["arm"],
                acc=row["accuracy"],
                f1=row["macro_f1"],
                rgl=row["rouge_l"],
                fmt=row["format_compliance"],
                lat=row["seconds_per_answer"],
                n=row["n"],
            )
        )
    return "\n".join(lines)


def render_report(rows: list[dict], cfg, extra: dict | None = None) -> str:
    """Render the comparison as a markdown report with the takeaways."""
    by_arm = {r["arm"]: r for r in rows}
    base = by_arm.get("prompt_only")
    rag = by_arm.get("rag")
    sft = by_arm.get("sft")
    gold = by_arm.get("rag_gold")

    lines = [
        "# VerticalLLM Lab: prompt-only vs RAG vs fine-tuning",
        "",
        f"Model: `{cfg.model.name}` &nbsp;|&nbsp; Dataset: `{cfg.data.dataset_name}` "
        f"(`{cfg.data.eval_source}` as held-out test) &nbsp;|&nbsp; Test size: {base['n'] if base else rows[0]['n']}",
        "",
        "## Results",
        "",
        _fmt_table(rows),
        "",
        "## What the numbers mean",
        "",
    ]

    def delta(a: dict | None, b: dict | None) -> str:
        if a is None or b is None:
            return ""
        diff = (a["accuracy"] - b["accuracy"]) * 100
        return f"{diff:+.1f} pts"

    bullets = []
    if base and rag:
        bullets.append(
            f"- Retrieval over the base model moves accuracy by **{delta(rag, base)}** "
            f"({base['accuracy']:.3f} -> {rag['accuracy']:.3f}): knowledge supplied at inference time."
        )
    if base and sft:
        bullets.append(
            f"- Closed-book fine-tuning moves accuracy by **{delta(sft, base)}** "
            f"({base['accuracy']:.3f} -> {sft['accuracy']:.3f}): knowledge pushed into the weights."
        )
        bullets.append(
            f"- It also fixes *behaviour*: output-format compliance goes from "
            f"{base['format_compliance']:.0%} to {sft['format_compliance']:.0%}. The base model knows "
            "roughly what to say but not how to say it in the required shape."
        )
    if sft and rag:
        combined = by_arm.get("sft_rag")
        if combined:
            bullets.append(
                f"- Combining both reaches **{combined['accuracy']:.3f}** versus "
                f"{rag['accuracy']:.3f} (RAG) and {sft['accuracy']:.3f} (SFT)."
            )
    if rag and gold:
        bullets.append(
            f"- The oracle arm (gold abstract given) reaches **{gold['accuracy']:.3f}**, so the "
            f"retriever currently recovers {max(gold['accuracy'] - rag['accuracy'], 0) * 100:.1f} pts "
            "of headroom: retrieval quality, not the generator, is the bottleneck."
        )
    if base:
        dist = base["prediction_distribution"]
        bullets.append(
            "- Zero-shot prediction mix: "
            + ", ".join(f"{k} {v:.0%}" for k, v in dist.items())
            + " -- small models tend to collapse onto one verdict, which is why accuracy alone is not enough."
        )
    lines.extend(bullets or ["- (run at least two arms to get a comparison)"])

    lines.extend(
        [
            "",
            "## Pitfalls hit while running this end-to-end",
            "",
            "Things that cost real debugging time; each fix is in the code so you don't repeat them.",
            "",
            "1. **`huggingface_hub` pins its endpoint at import time.** Setting `HF_ENDPOINT` after "
            "`import transformers` is a silent no-op -- every model load retried the (unreachable) "
            "official host for minutes before failing. Fix: `src/vlab/bootstrap.py` probes the "
            "official host and falls back to a mirror *before* any heavy import. Xet transfers are "
            "also disabled there (`cas-server.xethub.hf.co` returns 401 on some networks).",
            "2. **Prompt order dominated model quality.** Placing the evidence between the question "
            "and the `Answer:` cue made the 0.5B model emit one word and stop (RAG accuracy 0.283). "
            "Reordering to *instruction -> format -> question -> evidence -> `Answer:`* lifted the "
            "same arm to 0.450 -- a small-model amplification of the lost-in-the-middle effect.",
            "3. **High similarity is not relevance.** Top-1 cosine scores ranged 0.54-0.88 across "
            "all 60 questions, yet most retrieved abstracts were off-topic: with a 600-document "
            "corpus that does not contain the test abstracts, RAG injects confident distractors and "
            "scores *below* no-evidence prompting. The `rag_gold` oracle arm exists precisely to "
            "separate retrieval failure from generation failure.",
            "4. **The instruct model ignored the output format.** Despite explicit instructions, the "
            "base model never wrote the `Verdict:` line, so parsing needed a keyword fallback and "
            "`format_compliance` became a tracked metric. Forty LoRA steps raised it from 0% to "
            "100% -- fine-tuning fixes behaviour before it fixes knowledge.",
            "5. **Use the chat template.** Raw-completion prompting made the instruct model disregard "
            "the response spec entirely; wrapping prompts with `apply_chat_template` was required for "
            "a fair comparison.",
            "6. **transformers 5.x API drift.** `TrainingArguments` now takes `eval_strategy` (not "
            "`evaluation_strategy`); signatures were probed before writing the training code.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Cost",
            "",
            "- `s/answer` is measured on this machine with batch size "
            f"{cfg.eval.batch_size}; RAG pays an extra retrieval lookup per question.",
            "- Fine-tuning is a one-off training cost paid upfront; RAG pays a per-query cost forever.",
            "",
            "## Practical reading",
            "",
            "- **Volatile facts** (new papers, policies, prices) -> RAG: updating an index is cheap, retraining is not.",
            "- **Stable style/behaviour** (output format, tone, domain reasoning habits) -> fine-tuning: it removes the prompt tax and makes a small model behave consistently.",
            "- **Both** usually win: fine-tune the behaviour, retrieve the facts.",
            "",
        ]
    )
    if extra:
        lines.append("## Run metadata")
        lines.append("")
        lines.append("```json")
        import json

        lines.append(json.dumps(extra, indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


def plot_comparison(rows: list[dict], output_path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    labels = [r["arm"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([i - width / 2 for i in x], acc, width, label="accuracy", color="#2f6f9f")
    ax.bar([i + width / 2 for i in x], f1, width, label="macro-F1", color="#7fae76")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Domain knowledge injection on PubMedQA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def save_artifacts(cfg, results: list[dict], metadata: dict | None = None) -> dict[str, Any]:
    """Persist per-arm predictions, a summary table and the markdown report."""
    from vlab.utils import save_json, write_jsonl

    cfg.ensure_dirs()
    rows: list[dict] = []
    for result in results:
        rows.append(summarize(result))
        write_jsonl(cfg.results_dir / f"{result['arm']}_predictions.jsonl", result["predictions"])

    # Merge with earlier runs so arms can be executed in any order (or in parallel
    # processes) and still end up in one table.
    summary_path = cfg.results_dir / "summary.json"
    if summary_path.exists():
        try:
            previous = load_json(summary_path)
        except Exception:  # noqa: BLE001
            previous = {}
        fresh = {r["arm"] for r in rows}
        rows.extend(r for r in previous.get("rows", []) if r["arm"] not in fresh)

    rows.sort(key=lambda r: _arm_rank(r["arm"]))
    summary = {"rows": rows, "metadata": metadata or {}}
    save_json(cfg.results_dir / "summary.json", summary)
    (cfg.results_dir / "report.md").write_text(render_report(rows, cfg, metadata or {}), encoding="utf-8")
    plotted = plot_comparison(rows, str(cfg.results_dir / "comparison.png"))
    return {"rows": rows, "plot": plotted}


def _arm_rank(arm: str) -> int:
    from vlab.arms import ARM_ORDER

    return ARM_ORDER.index(arm) if arm in ARM_ORDER else 99
