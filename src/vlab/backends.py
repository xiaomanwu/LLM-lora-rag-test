"""Model loading and batched text generation.

The same :class:`Generator` is reused by all four arms, so token budget, decoding
strategy and prompt formatting stay identical across the comparison.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlab.utils import get_logger

LOGGER = get_logger("vlab.backends")


def _dtype(name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(cfg, adapter_path: str | None = None):
    """Load the base model, optionally with a LoRA adapter on top.

    The adapter is kept separate (not merged) so the very same base weights back
    the ``prompt_only``/``rag`` arms and the ``sft``/``sft_rag`` arms.
    """
    from vlab.config import setup_hf_endpoint

    setup_hf_endpoint(cfg.hf_endpoint)
    dtype = _dtype(cfg.model.dtype)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)
    model.config.use_cache = True
    # Pretrained generation configs often carry sampling flags (top_k/top_p) that
    # are meaningless under greedy decoding and trigger warnings.
    if hasattr(model, "generation_config"):
        model.generation_config.top_k = None
        model.generation_config.top_p = None

    if adapter_path:
        from peft import PeftModel

        LOGGER.info("attaching LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model


class Generator:
    """Greedy batched generation for short structured answers."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 96, temperature: float = 0.0,
                 top_p: float = 1.0, batch_size: int = 8, device: str = "cpu",
                 use_chat_template: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.device = device
        self.use_chat_template = use_chat_template and getattr(tokenizer, "chat_template", None) is not None

    def _wrap(self, prompts: list[str]) -> list[str]:
        if not self.use_chat_template:
            return list(prompts)
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

    def generate(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = self._wrap(prompts[start : start + self.batch_size])
            encoded = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                                     max_length=2048).to(self.model.device)
            do_sample = self.temperature > 0
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=self.top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            for seq in generated:
                outputs.append(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip())
        return outputs

    def timed_generate(self, prompts: list[str]) -> tuple[list[str], float]:
        start = time.perf_counter()
        outputs = self.generate(prompts)
        return outputs, time.perf_counter() - start


def build_generator(cfg, model=None, adapter_path: str | None = None, batch_size: int | None = None) -> Generator:
    tokenizer = load_tokenizer(cfg.model.name)
    if model is None:
        model = load_model(cfg, adapter_path=adapter_path)
    return Generator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.eval.max_new_tokens or cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        batch_size=batch_size or cfg.eval.batch_size,
        device=cfg.device,
        use_chat_template=cfg.model.use_chat_template,
    )
