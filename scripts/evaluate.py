#!/usr/bin/env python3
"""Run the experiment arms and write the comparison report.

Examples
--------
    python3 scripts/evaluate.py --arms prompt_only rag
    python3 scripts/evaluate.py --arms prompt_only rag sft sft_rag rag_gold --limit 200
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vlab.bootstrap as bootstrap  # noqa: E402

bootstrap.setup_environment()

from vlab.arms import ARM_ORDER, RETRIEVAL_ARMS, run_arm  # noqa: E402
from vlab.backends import build_generator  # noqa: E402
from vlab.config import Config, resolve_device, setup_hf_endpoint  # noqa: E402
from vlab.data.corpus import Retriever  # noqa: E402
from vlab.evaluate import save_artifacts  # noqa: E402
from vlab.utils import get_logger, read_jsonl, seed_everything  # noqa: E402

LOGGER = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--arms", nargs="+", default=["prompt_only", "rag"], choices=ARM_ORDER)
    parser.add_argument("--adapter", default="", help="path to the LoRA adapter directory")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N test records")
    parser.add_argument("--few-shot", type=int, default=0, help="prepend N worked examples to the prompt")
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    if args.hf_endpoint:
        cfg.hf_endpoint = args.hf_endpoint
    cfg.device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    cfg.ensure_dirs()
    if resolved := setup_hf_endpoint(cfg.hf_endpoint):
        LOGGER.info("using HF endpoint %s", resolved)

    test_rows = read_jsonl(cfg.data_dir / "test.jsonl")
    if not test_rows:
        raise FileNotFoundError(f"no test data at {cfg.data_dir / 'test.jsonl'}; run scripts/prepare_data.py")
    limit = args.limit or cfg.data.max_eval_samples
    if limit:
        test_rows = test_rows[:limit]
    LOGGER.info("evaluating on %s test records", len(test_rows))

    needs_retrieval = any(a in RETRIEVAL_ARMS and a != "rag_gold" for a in args.arms)
    retriever = None
    if needs_retrieval:
        retriever = Retriever.load(cfg.data_dir / "index", cfg.retriever.embedding_model, cfg.device)
        LOGGER.info("loaded index with %s documents", len(retriever.documents))

    results = []
    base_arms = [a for a in args.arms if a in ("prompt_only", "rag", "rag_gold")]
    ft_arms = [a for a in args.arms if a in ("sft", "sft_rag")]

    if base_arms:
        generator = build_generator(cfg)
        for arm in base_arms:
            results.append(
                run_arm(cfg, arm, test_rows, retriever, generator=generator, few_shot=args.few_shot)
            )
        del generator
        gc.collect()

    if ft_arms:
        adapter = args.adapter or str(cfg.checkpoint_dir / "sft_adapter")
        if not Path(adapter).exists():
            raise FileNotFoundError(f"adapter not found at {adapter}; run scripts/train_sft.py first")
        generator = build_generator(cfg, adapter_path=adapter)
        for arm in ft_arms:
            results.append(
                run_arm(cfg, arm, test_rows, retriever, adapter_path=adapter, generator=generator,
                        few_shot=args.few_shot)
            )

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": cfg.device,
        "arms": args.arms,
        "test_size": len(test_rows),
        "config": cfg.to_dict(),
    }
    artifacts = save_artifacts(cfg, results, metadata)
    for row in artifacts["rows"]:
        LOGGER.info(
            "%-11s acc=%.3f macro_f1=%.3f rouge_l=%.3f (%.2fs/answer)",
            row["arm"], row["accuracy"], row["macro_f1"], row["rouge_l"], row["seconds_per_answer"],
        )
    LOGGER.info("report -> %s", cfg.results_dir / "report.md")


if __name__ == "__main__":
    main()
