# VerticalLLM Lab: prompt-only vs RAG vs fine-tuning

Model: `Qwen/Qwen2.5-0.5B-Instruct` &nbsp;|&nbsp; Dataset: `qiaojin/PubMedQA` (`pqa_labeled` as held-out test) &nbsp;|&nbsp; Test size: 60

## Results

| arm | accuracy | macro_f1 | rouge_l | format% | s/answer | n |
|---|---|---|---|---|---|---|
| prompt_only | 0.617 | 0.363 | 0.182 | 0% | 16.55 | 60 |
| rag | 0.450 | 0.319 | 0.048 | 0% | 11.38 | 60 |
| sft | 0.567 | 0.241 | 0.183 | 100% | 17.56 | 60 |
| sft_rag | 0.567 | 0.241 | 0.160 | 100% | 16.63 | 60 |
| rag_gold | 0.683 | 0.472 | 0.132 | 0% | 15.45 | 60 |

## What the numbers mean

- Retrieval over the base model moves accuracy by **-16.7 pts** (0.617 -> 0.450): knowledge supplied at inference time.
- Closed-book fine-tuning moves accuracy by **-5.0 pts** (0.617 -> 0.567): knowledge pushed into the weights.
- It also fixes *behaviour*: output-format compliance goes from 0% to 100%. The base model knows roughly what to say but not how to say it in the required shape.
- Combining both reaches **0.567** versus 0.450 (RAG) and 0.567 (SFT).
- The oracle arm (gold abstract given) reaches **0.683**, so the retriever currently recovers 23.3 pts of headroom: retrieval quality, not the generator, is the bottleneck.
- Zero-shot prediction mix: yes 92%, no 8%, maybe 0% -- small models tend to collapse onto one verdict, which is why accuracy alone is not enough.

## Pitfalls hit while running this end-to-end

Things that cost real debugging time; each fix is in the code so you don't repeat them.

1. **`huggingface_hub` pins its endpoint at import time.** Setting `HF_ENDPOINT` after `import transformers` is a silent no-op -- every model load retried the (unreachable) official host for minutes before failing. Fix: `src/vlab/bootstrap.py` probes the official host and falls back to a mirror *before* any heavy import. Xet transfers are also disabled there (`cas-server.xethub.hf.co` returns 401 on some networks).
2. **Prompt order dominated model quality.** Placing the evidence between the question and the `Answer:` cue made the 0.5B model emit one word and stop (RAG accuracy 0.283). Reordering to *instruction -> format -> question -> evidence -> `Answer:`* lifted the same arm to 0.450 -- a small-model amplification of the lost-in-the-middle effect.
3. **High similarity is not relevance.** Top-1 cosine scores ranged 0.54-0.88 across all 60 questions, yet most retrieved abstracts were off-topic: with a 600-document corpus that does not contain the test abstracts, RAG injects confident distractors and scores *below* no-evidence prompting. The `rag_gold` oracle arm exists precisely to separate retrieval failure from generation failure.
4. **The instruct model ignored the output format.** Despite explicit instructions, the base model never wrote the `Verdict:` line, so parsing needed a keyword fallback and `format_compliance` became a tracked metric. Forty LoRA steps raised it from 0% to 100% -- fine-tuning fixes behaviour before it fixes knowledge.
5. **Use the chat template.** Raw-completion prompting made the instruct model disregard the response spec entirely; wrapping prompts with `apply_chat_template` was required for a fair comparison.
6. **transformers 5.x API drift.** `TrainingArguments` now takes `eval_strategy` (not `evaluation_strategy`); signatures were probed before writing the training code.


## Cost

- `s/answer` is measured on this machine with batch size 16; RAG pays an extra retrieval lookup per question.
- Fine-tuning is a one-off training cost paid upfront; RAG pays a per-query cost forever.

## Practical reading

- **Volatile facts** (new papers, policies, prices) -> RAG: updating an index is cheap, retraining is not.
- **Stable style/behaviour** (output format, tone, domain reasoning habits) -> fine-tuning: it removes the prompt tax and makes a small model behave consistently.
- **Both** usually win: fine-tune the behaviour, retrieve the facts.

## Run metadata

```json
{
  "generated_at": "2026-10-28 00:00:19",
  "device": "cpu",
  "arms": [
    "sft_rag"
  ],
  "test_size": 60,
  "config": {
    "seed": 42,
    "device": "cpu",
    "hf_endpoint": "https://hf-mirror.com",
    "paths": {
      "root": ".",
      "data": "data",
      "checkpoints": "checkpoints",
      "results": "results"
    },
    "data": {
      "dataset_name": "qiaojin/PubMedQA",
      "train_source": "pqa_artificial",
      "eval_source": "pqa_labeled",
      "max_train_samples": 1200,
      "max_eval_samples": 60,
      "corpus_size": 600,
      "dev_size": 100,
      "seed": 42
    },
    "model": {
      "name": "Qwen/Qwen2.5-0.5B-Instruct",
      "max_new_tokens": 64,
      "temperature": 0.0,
      "top_p": 1.0,
      "dtype": "float32",
      "attn_implementation": "sdpa",
      "use_chat_template": true
    },
    "retriever": {
      "embedding_model": "BAAI/bge-small-en-v1.5",
      "top_k": 2,
      "batch_size": 32,
      "normalize": true,
      "exclude_gold_documents": true,
      "max_doc_chars": 700
    },
    "sft": {
      "lora_r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
      ],
      "train_with_context": false,
      "num_train_epochs": 1.0,
      "per_device_train_batch_size": 2,
      "gradient_accumulation_steps": 2,
      "learning_rate": 0.0002,
      "weight_decay": 0.0,
      "warmup_ratio": 0.1,
      "lr_scheduler_type": "cosine",
      "max_length": 384,
      "logging_steps": 5,
      "save_total_limit": 1,
      "eval_fraction": 0.0,
      "max_steps": 40
    },
    "eval": {
      "batch_size": 16,
      "max_new_tokens": 64
    }
  }
}
```
