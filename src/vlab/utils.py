"""Small helpers: seeding, logging, jsonl IO and timing."""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "vlab", level: int = logging.INFO) -> logging.Logger:
    if logging.getLogger(name).handlers:
        return logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    start = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        yield stats
    finally:
        stats["seconds"] = round(time.perf_counter() - start, 3)
        if logger is not None:
            logger.info("%s took %.2fs", label, stats["seconds"])
