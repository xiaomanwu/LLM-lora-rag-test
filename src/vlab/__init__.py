"""VerticalLLM Lab: a compact benchmark for domain adaptation of small LLMs.

The package compares three ways of injecting vertical-domain knowledge into a
small open-weight LLM, all measured on the same held-out test split:

1. ``prompt_only``  -- zero-shot prompting, the model relies on parametric memory.
2. ``rag``          -- retrieval-augmented generation over a domain corpus.
3. ``sft``          -- supervised fine-tuning (LoRA) that pushes knowledge into parameters.
4. ``sft_rag``      -- fine-tuned model + retrieval (the combination arm).
"""

__version__ = "0.1.0"
