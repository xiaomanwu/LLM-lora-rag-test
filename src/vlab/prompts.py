"""Prompt templates shared by every experiment arm.

Keeping the templates in one place matters for a fair comparison: the four arms
differ only in *what context they get*, not in how the question is phrased or
how the answer is requested.
"""

from __future__ import annotations

VERDICTS = ("yes", "no", "maybe")

INSTRUCTION = (
    "You are a biomedical research assistant. Answer the research question with a "
    "short verdict followed by a one-sentence conclusion."
)

OUTPUT_FORMAT = (
    "Respond in exactly two lines:\n"
    "Verdict: <yes|no|maybe>\n"
    "Conclusion: <one sentence>"
)


def format_evidence(documents: list[str]) -> str:
    """Render retrieved abstracts as a numbered evidence block."""
    if not documents:
        return ""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(f"[{i}] {doc.strip()}")
    return "Evidence:\n" + "\n".join(blocks)


def build_few_shot_block(examples: list[dict[str, str]] | None) -> str:
    if not examples:
        return ""
    parts = []
    for ex in examples:
        parts.append(
            f"Question: {ex['question']}\n"
            f"Verdict: {ex['verdict']}\n"
            f"Conclusion: {ex['conclusion']}"
        )
    return "Here are some examples.\n\n" + "\n\n".join(parts) + "\n"


def prompt_only(question: str, few_shot: list[dict[str, str]] | None = None) -> str:
    """Zero-shot (or few-shot) prompt: parametric memory only."""
    pieces = [INSTRUCTION, OUTPUT_FORMAT]
    block = build_few_shot_block(few_shot)
    if block:
        pieces.append(block)
    pieces.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(p for p in pieces if p)


def rag_prompt(question: str, documents: list[str], few_shot: list[dict[str, str]] | None = None) -> str:
    """Retrieval-augmented prompt: the same question plus retrieved evidence.

    The question is placed *before* the evidence and the answer cue is kept last.
    Small models lose the thread when a long evidence block separates the
    question from the `Answer:` cue — a variant of the lost-in-the-middle effect.
    """
    pieces = [INSTRUCTION, OUTPUT_FORMAT]
    block = build_few_shot_block(few_shot)
    if block:
        pieces.append(block)
    pieces.append(f"Question: {question}")
    pieces.append("Use the abstracts below as evidence.")
    pieces.append(format_evidence(documents))
    pieces.append("Answer:")
    return "\n\n".join(p for p in pieces if p)


def build_prompt(question: str, documents: list[str] | None = None) -> str:
    if documents:
        return rag_prompt(question, documents)
    return prompt_only(question)


# --------------------------------------------------------------------- parse
def parse_prediction(text: str) -> tuple[str, str]:
    """Extract (verdict, conclusion) from a raw generation.

    Models are small and sometimes improvise, so parsing is defensive: it scans
    for the first ``Verdict:``/``Conclusion:`` markers and falls back to keyword
    matching over the whole output.
    """
    verdict, conclusion = "", ""
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if not verdict and lower.startswith("verdict:"):
            token = stripped.split(":", 1)[1].strip().strip(".").lower()
            if token in VERDICTS:
                verdict = token
            else:
                for v in VERDICTS:
                    if v in token:
                        verdict = v
                        break
        elif not conclusion and lower.startswith("conclusion:"):
            conclusion = stripped.split(":", 1)[1].strip()
    if not verdict:
        lower = text.lower()
        hits = [(lower.find(v), v) for v in VERDICTS if v in lower]
        if hits:
            verdict = min(hits)[1]
    if not conclusion:
        conclusion = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return verdict or "maybe", conclusion


def format_target(verdict: str, conclusion: str) -> str:
    return f"Verdict: {verdict}\nConclusion: {conclusion}"
