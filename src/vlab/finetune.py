"""LoRA supervised fine-tuning of the base model on PubMedQA.

Two training regimes are supported (``sft.train_with_context``):

* ``False`` (default) -- *closed-book*: the model only sees the question and must
  produce the verdict + conclusion. Whatever it learns is stored in the weights,
  which makes this arm directly comparable to ``prompt_only``.
* ``True``  -- the model also sees the abstract, i.e. it is taught the
  "read the evidence, then answer" behaviour used by the RAG arm.

Loss is computed on the completion tokens only; prompt tokens are masked out with
``-100`` so the model is not trained to imitate the instruction template.
"""

from __future__ import annotations

import inspect
import random

import torch
from torch.utils.data import Dataset

from vlab import prompts
from vlab.utils import get_logger, read_jsonl

LOGGER = get_logger("vlab.finetune")


def build_examples(cfg) -> list[dict]:
    rows = read_jsonl(cfg.data_dir / "train.jsonl")
    examples = []
    for row in rows:
        conclusion = (row.get("conclusion") or "").strip()
        if not conclusion:
            continue
        target = prompts.format_target(row["verdict"], conclusion)
        if cfg.sft.train_with_context:
            prompt = prompts.rag_prompt(row["question"], [row["context"]])
        else:
            prompt = prompts.prompt_only(row["question"])
        examples.append({"prompt": prompt, "completion": target})
    random.Random(cfg.seed).shuffle(examples)
    LOGGER.info("built %s SFT examples (with_context=%s)", len(examples), cfg.sft.train_with_context)
    return examples


class SFTDataset(Dataset):
    """Prompt/completion pairs tokenized once and masked on the prompt side."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int,
                 use_chat_template: bool = False) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template and getattr(tokenizer, "chat_template", None) is not None
        self.items = [self._encode(e) for e in examples]

    def _encode(self, example: dict) -> dict:
        prompt_text = example["prompt"] + "\n"
        if self.use_chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
        completion_ids = completion_ids + [self.tokenizer.eos_token_id]
        ids = (prompt_ids + completion_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_length]
        return {"input_ids": ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class Collator:
    """Right-pad a batch and mask padding tokens out of the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention.append([1] * len(f["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def _make_trainer(model, tokenizer, train_dataset, collator, cfg, output_dir: str):
    from transformers import Trainer, TrainingArguments

    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = cfg.device == "cuda" and bool(supports_bf16 and supports_bf16())
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.sft.num_train_epochs,
        max_steps=cfg.sft.max_steps if cfg.sft.max_steps > 0 else -1,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        weight_decay=cfg.sft.weight_decay,
        warmup_ratio=cfg.sft.warmup_ratio,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        logging_steps=cfg.sft.logging_steps,
        save_strategy="steps" if cfg.sft.max_steps > 0 else "no",
        save_steps=max(1, cfg.sft.max_steps // 2) if cfg.sft.max_steps > 0 else 500,
        save_total_limit=cfg.sft.save_total_limit,
        bf16=use_bf16,
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        seed=cfg.seed,
        dataloader_num_workers=2,
    )
    kwargs: dict = {}
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        **kwargs,
    )


def train(cfg) -> str:
    """Run LoRA SFT and return the directory holding the adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from vlab.backends import load_tokenizer
    from vlab.config import setup_hf_endpoint

    setup_hf_endpoint(cfg.hf_endpoint)
    cfg.ensure_dirs()

    tokenizer = load_tokenizer(cfg.model.name)
    examples = build_examples(cfg)
    if not examples:
        raise FileNotFoundError("no SFT examples; run scripts/prepare_data.py first")

    dataset = SFTDataset(examples, tokenizer, cfg.sft.max_length, cfg.model.use_chat_template)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype={"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.model.dtype],
        attn_implementation=cfg.model.attn_implementation,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.sft.lora_r,
        lora_alpha=cfg.sft.lora_alpha,
        lora_dropout=cfg.sft.lora_dropout,
        target_modules=list(cfg.sft.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output_dir = str(cfg.checkpoint_dir / "sft")
    trainer = _make_trainer(model, tokenizer, dataset, Collator(tokenizer.pad_token_id), cfg, output_dir)
    trainer.train()

    adapter_dir = str(cfg.checkpoint_dir / "sft_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    LOGGER.info("adapter saved to %s", adapter_dir)
    return adapter_dir
