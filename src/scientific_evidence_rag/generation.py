"""CPU-only, pinned FLAN-T5 generation with strict evidence-context prompts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from scientific_evidence_rag.paths import CACHE_DIR

MODEL_ID = "google/flan-t5-small"
MODEL_REVISION = "9471d3bc4f85c9012776f03c4c00fdfe0d789a95"
VERDICTS = ("SUPPORTS", "REFUTES", "MIXED", "NOT_ENOUGH_EVIDENCE")


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    verdict: str | None
    cited_doc_ids: tuple[str, ...]


def parse_verdict(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.upper()).strip()
    match = re.match(
        r"^(?:VERDICT\s*[:=-]\s*)?("
        r"NOT[_ ]+ENOUGH[_ ]+EVIDENCE|INSUFFICIENT EVIDENCE|MIXED|REFUTES?|"
        r"CONTRADICTS?|SUPPORTS?)\b",
        normalized,
    )
    if not match:
        return None
    label = match.group(1)
    if re.fullmatch(r"NOT[_ ]+ENOUGH[_ ]+EVIDENCE|INSUFFICIENT EVIDENCE", label):
        return "NOT_ENOUGH_EVIDENCE"
    if label == "MIXED":
        return "MIXED"
    if label in {"REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTES"
    if label in {"SUPPORT", "SUPPORTS"}:
        return "SUPPORTS"
    return None


def parse_citations(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\[(\d+)\]", text)))


class FlanT5Generator:
    def __init__(self, max_input_tokens: int = 512, max_new_tokens: int = 96) -> None:
        if max_input_tokens < 128 or max_input_tokens > 1024:
            raise ValueError("max_input_tokens must be in [128, 1024]")
        if max_new_tokens < 16 or max_new_tokens > 192:
            raise ValueError("max_new_tokens must be in [16, 192]")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Local generation requires `make install-llm`.") from error

        torch.set_num_threads(max(1, min(int(os.getenv("RAG_TORCH_THREADS", "2")), 4)))
        cache = CACHE_DIR / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache),
            trust_remote_code=False,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache),
            trust_remote_code=False,
        )
        self._model.to("cpu")
        self._model.eval()
        self._torch = torch

    def generate(self, claim: str, evidence: list[dict[str, object]]) -> GeneratedAnswer:
        self._load()
        if self._tokenizer is None or self._model is None or self._torch is None:
            raise RuntimeError("The local generation model did not load")
        context: list[str] = []
        for item in evidence[:3]:
            doc_id = str(item["doc_id"])
            title = str(item["title"])
            sentences = item.get("abstract", [])
            excerpt = " ".join(str(sentence) for sentence in sentences[:3])
            context.append(f"[{doc_id}] {title}. {excerpt}")
        prompt = (
            "Using only the evidence below, decide whether it supports or refutes the claim, "
            "whether evidence conflicts, or whether the evidence is insufficient. "
            "Reply with exactly one label and nothing else: SUPPORTS, REFUTES, MIXED, "
            "or NOT_ENOUGH_EVIDENCE. Do not repeat the claim. "
            "This benchmark is not medical advice.\n\n"
            f"CLAIM: {claim}\n\nEVIDENCE:\n" + "\n".join(context) + "\n\nVERDICT:"
        )
        batch = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        with self._torch.inference_mode():
            output = self._model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return GeneratedAnswer(text, parse_verdict(text), parse_citations(text))
