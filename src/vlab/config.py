"""Typed configuration for VerticalLLM Lab.

Configuration is expressed as nested dataclasses so that every experiment arm
shares one source of truth. Values can be overridden from a YAML file or from
individual CLI flags (``--set data.max_eval_samples=100``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Some networks cannot reach huggingface.co directly. Allow a drop-in mirror.
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")


def setup_hf_endpoint(endpoint: str | None = None) -> str:
    """Make sure a reachable HF host is selected; returns the endpoint in use.

    The real work lives in ``vlab.bootstrap``, which must run before
    transformers is imported — the endpoint is frozen at import time. This
    wrapper exists so library callers can pass an explicit mirror.
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        return os.environ["HF_ENDPOINT"]
    from vlab import bootstrap

    return bootstrap.setup_environment()


@dataclass
class PathConfig:
    root: str = "."
    data: str = "data"
    checkpoints: str = "checkpoints"
    results: str = "results"


@dataclass
class DataConfig:
    """Everything related to the vertical-domain dataset.

    ``eval_source`` is the human-annotated split used as the held-out test set.
    ``train_source`` is the (auto-labelled) split used for SFT and as the
    retrieval corpus. Documents belonging to the test split are removed from the
    corpus so retrieval can never leak the gold abstract.
    """

    dataset_name: str = "qiaojin/PubMedQA"
    train_source: str = "pqa_artificial"
    eval_source: str = "pqa_labeled"
    max_train_samples: int = 20000
    max_eval_samples: int = 500
    corpus_size: int = 50000
    dev_size: int = 100
    seed: int = 42


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0
    dtype: str = "float32"
    attn_implementation: str = "sdpa"
    # Instruction-tuned models are trained with a chat template; feeding raw text
    # instead measurably hurts their ability to follow an output format.
    use_chat_template: bool = True


@dataclass
class RetrieverConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 3
    batch_size: int = 64
    normalize: bool = True
    exclude_gold_documents: bool = True
    # Abstracts are long; cap each one so the evidence block stays readable and
    # cheap to prefill. This also mirrors what production RAG systems do.
    max_doc_chars: int = 1200


@dataclass
class SFTConfig:
    """LoRA supervised fine-tuning hyper-parameters.

    ``train_with_context`` switches between two training regimes:
      * False -> closed-book SFT (question -> answer), the pure "knowledge into
        parameters" arm, directly comparable with ``prompt_only``.
      * True  -> the model is trained to read retrieved abstracts first.
    """

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    train_with_context: bool = False
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_length: int = 512
    logging_steps: int = 10
    save_total_limit: int = 1
    eval_fraction: float = 0.05
    max_steps: int = -1


@dataclass
class EvalConfig:
    batch_size: int = 8
    max_new_tokens: int = 96


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    hf_endpoint: str = ""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        cfg = cls()
        if path is not None and Path(path).exists():
            raw = yaml.safe_load(Path(path).read_text()) or {}
            cfg = cls.from_dict(raw)
        for item in overrides or []:
            key, value = item.split("=", 1)
            cfg.set(key, _coerce(value))
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        base = cls()
        for section, values in raw.items():
            if not isinstance(values, dict):
                continue
            target = getattr(base, section, None)
            if target is None:
                continue
            for k, v in values.items():
                if hasattr(target, k):
                    setattr(target, k, v)
            setattr(base, section, target)
        for k in ("seed", "device", "hf_endpoint"):
            if k in raw:
                setattr(base, k, raw[k])
        return base

    def set(self, dotted_key: str, value: Any) -> None:
        section, _, key = dotted_key.partition(".")
        if not key:
            setattr(self, dotted_key, value)
            return
        target = getattr(self, section)
        if not hasattr(target, key):
            raise KeyError(f"unknown config key: {dotted_key}")
        setattr(target, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --------------------------------------------------------------- paths
    def resolve(self, relative: str) -> Path:
        return Path(self.paths.root) / relative

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data)

    @property
    def checkpoint_dir(self) -> Path:
        return self.resolve(self.paths.checkpoints)

    @property
    def results_dir(self) -> Path:
        return self.resolve(self.paths.results)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.checkpoint_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


def _coerce(value: str) -> Any:
    """Turn CLI strings into python literals (ints, floats, bools, lists)."""
    lowered = value.strip().lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    return value


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference in ("cpu", "cuda"):
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"
