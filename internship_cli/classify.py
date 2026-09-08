"""Optional company classification via an LLM (Gemini or local Ollama).

Read this before trusting any output of this module.

An LLM has no live view of funding databases. It works from a fixed training
cutoff, so a round closed last month is invisible to it and a company that has
since died may still look "funded". Accuracy is worst exactly where this tool
needs help most: small, obscure, name-colliding firms.

Therefore:
  * every verdict is tagged `<backend>-guess` and carries a confidence,
  * "unknown" is an allowed and encouraged answer,
  * the model is never asked for a funding round, amount, or date, because a
    fabricated "Series A" would read like a fact.

Use `--allowlist` when you need a filter you can defend. Use this to triage.

Backend selection (env):
  LLM_BACKEND=gemini|ollama   (default gemini)
  OLLAMA_MODEL=qwen2.5:3b     (when LLM_BACKEND=ollama; also readable from .env)
  GEMINI_API_KEY=...          (when LLM_BACKEND=gemini)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from .llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    GEMINI_API_KEY_ENV,
    LLMError,
    Message,
    OLLAMA_MODEL_ENV,
    get_backend,
    get_llm_client,
)

DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
API_KEY_ENV = GEMINI_API_KEY_ENV

CATEGORIES = (
    "funded_startup",
    "big_tech_or_enterprise",
    "agency_or_consultancy",
    "education_or_training",
    "unknown",
)

STARTUP_CATEGORIES = ("funded_startup",)

_PROMPT = """You are labelling company names scraped from LinkedIn hiring posts.

For each company, choose exactly one category:
- "funded_startup": a venture-backed startup you specifically recognise.
- "big_tech_or_enterprise": a large established company or enterprise.
- "agency_or_consultancy": a staffing agency, recruiter, IT services or outsourcing firm.
- "education_or_training": an edtech, coaching, bootcamp or training provider.
- "unknown": you do not specifically recognise this company.

Critical rules:
1. If you do not genuinely recognise the specific company, answer "unknown".
   Most of these are small and obscure. "unknown" is the correct, expected
   answer for many of them and is far more useful than a guess.
2. Do NOT infer from the name. A name containing "AI", "Labs" or "Tech" tells
   you nothing about funding.
3. Do NOT output funding rounds, amounts, investors or dates. You cannot know
   these reliably and inventing them is worse than useless.
4. confidence: "high" only for companies you are certain about, otherwise
   "medium" or "low".

Return JSON only, shaped as:
{"results": [{"company": "<name as given>", "category": "<category>", "confidence": "high|medium|low"}]}

Companies:
"""


@dataclass(frozen=True)
class Classification:
    company: str
    category: str
    confidence: str
    source: str = "gemini-guess"
    verified: bool = False

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "verified": self.verified,
        }

    def label(self) -> str:
        return f"{self.category} ({self.confidence}, guess)"


class ClassificationError(RuntimeError):
    pass


def api_key_available() -> bool:
    """True when the configured backend can run without further setup."""
    if get_backend() == "ollama":
        return True
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def _resolve_model(model: Optional[str], backend: str) -> Optional[str]:
    """Pick the model for this backend.

    For ollama, OLLAMA_MODEL wins over a Gemini CLI default so switching
    backends via env alone is enough.
    """
    if backend == "ollama":
        env_model = os.environ.get(OLLAMA_MODEL_ENV, "").strip()
        if env_model:
            return env_model
        if model and model != DEFAULT_GEMINI_MODEL:
            return model
        return DEFAULT_OLLAMA_MODEL
    return model or DEFAULT_GEMINI_MODEL


class CompanyClassifier:
    """Batches company names to the configured LLM and caches the verdicts."""

    def __init__(self, model: Optional[str] = None, batch_size: int = 25) -> None:
        self.backend = get_backend()
        try:
            self._client = get_llm_client(_resolve_model(model, self.backend))
        except LLMError as exc:
            raise ClassificationError(str(exc)) from exc
        self.model = self._client.model
        self.batch_size = max(1, batch_size)
        self._source = f"{self.backend}-guess"
        self._cache: dict[str, Classification] = {}

    def classify(self, companies: Iterable[str]) -> dict[str, Classification]:
        unique = [c for c in dict.fromkeys(c.strip() for c in companies if c and c.strip())]
        pending = [c for c in unique if c.lower() not in self._cache]

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            self._classify_batch(batch)

        # Anything the model omitted stays explicitly unknown rather than absent.
        for company in unique:
            self._cache.setdefault(
                company.lower(),
                Classification(company, "unknown", "low", source=self._source),
            )

        return {c: self._cache[c.lower()] for c in unique}

    def _classify_batch(self, batch: list[str]) -> None:
        prompt = _PROMPT + "\n".join(f"- {name}" for name in batch)
        try:
            completion = self._client.complete(
                [Message(role="user", content=prompt)],
                temperature=0.0,
                json_mode=True,
            )
        except LLMError as exc:
            raise ClassificationError(str(exc)) from exc

        text = (completion.content or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            for name in batch:
                self._cache[name.lower()] = Classification(
                    name, "unknown", "low", source=self._source
                )
            return

        rows = parsed.get("results", []) if isinstance(parsed, dict) else parsed
        by_name = {str(name).lower(): name for name in batch}

        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("company", "")).strip()
            category = str(row.get("category", "unknown")).strip().lower()
            confidence = str(row.get("confidence", "low")).strip().lower()

            if category not in CATEGORIES:
                category = "unknown"
            if confidence not in ("high", "medium", "low"):
                confidence = "low"

            original = by_name.get(name.lower(), name)
            if original:
                self._cache[original.lower()] = Classification(
                    original, category, confidence, source=self._source
                )


def is_startup(classification: Classification | None) -> bool:
    """Whether a verdict counts as a startup for filtering purposes.

    "unknown" is deliberately NOT a startup: when the model does not recognise a
    company, dropping it is a filtering choice the user opted into via
    --llm-filter, and pretending otherwise would hide the uncertainty.
    """
    return bool(classification and classification.category in STARTUP_CATEGORIES)
