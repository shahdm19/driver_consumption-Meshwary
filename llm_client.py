"""Low-level LLM client: Groq primary, Gemini fallback.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH FOR THE PRIMARY→FALLBACK PATTERN.

The original code implemented the "call Groq, on failure fall back to Gemini"
pattern TWICE — once in ``extract_specs_with_llm`` (for JSON extraction) and
once in ``get_recommendations`` (for free-text).  Both copies are gone; the
pattern now lives in exactly one method: :meth:`LLMClient._call_with_fallback`.

Public API (used by every call site):
    - :meth:`LLMClient.extract_json`   — Groq→Gemini fallback for JSON output.
    - :meth:`LLMClient.generate_text`  — Groq→Gemini fallback for free-text.

Call sites (must NOT re-implement fallback):
    - llm_extraction.extract_specs_with_llm   -> llm_client.extract_json(...)
    - recommendations.get_recommendations     -> llm_client.generate_text(...)

Both public methods delegate to :meth:`_call_with_fallback`, which is the only
place the try/except + Gemini-availability check + logging lives.
"""
from __future__ import annotations

from typing import Callable, Optional

from config import (
    GEMINI_API_KEY,
    GEMINI_JSON_TEMPERATURE,
    GEMINI_MODEL,
    GEMINI_RECOMMENDATIONS_MAX_TOKENS,
    GEMINI_RECOMMENDATIONS_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_JSON_TEMPERATURE,
    GROQ_MODEL,
    GROQ_RECOMMENDATIONS_MAX_TOKENS,
    GROQ_RECOMMENDATIONS_TEMPERATURE,
    RETRY_POLICY,
)
from logging_config import get_logger

logger = get_logger(__name__)

# Lazily imported so a missing package does not crash startup.
try:
    from groq import Groq
    GROQ_AVAILABLE: bool = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_IMPORT_AVAILABLE: bool = True
except ImportError:
    GEMINI_IMPORT_AVAILABLE = False


class LLMClient:
    """Wraps Groq (primary) and Gemini (fallback) clients.

    Constructed once in ``main.py`` and passed explicitly to every caller
    (no global instance).
    """

    def __init__(self) -> None:
        self.groq = None
        self.gemini_model = None
        self.gemini_available: bool = False

        # ---- Groq (primary) ----
        if GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized")
        else:
            logger.warning("Groq client not initialized (missing key or library)")

        # ---- Gemini (fallback) ----
        if GEMINI_IMPORT_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                self.gemini_available = True
                logger.info(" Gemini fallback initialized (%s)", GEMINI_MODEL)
            except Exception as e:
                logger.warning(" Gemini init failed: %s", e)
        elif not GEMINI_IMPORT_AVAILABLE:
            logger.warning(
                " google-generativeai not installed. Gemini fallback disabled."
            )
            logger.warning("   Install with: pip install google-generativeai")

    # ------------------------------------------------------------------ #
    # Low-level primitives (each decorated with the shared RETRY_POLICY)  #
    # ------------------------------------------------------------------ #

    @RETRY_POLICY
    def _groq_json(self, system_prompt: str, user_msg: str) -> str:
        """Groq call for JSON extraction. Returns the raw content string."""
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=GROQ_JSON_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    @RETRY_POLICY
    def _groq_text(self, prompt: str) -> str:
        """Groq call for free-text (recommendations). Returns content string."""
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=GROQ_RECOMMENDATIONS_MAX_TOKENS,
            temperature=GROQ_RECOMMENDATIONS_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    @RETRY_POLICY
    def _gemini_json(self, system_prompt: str, user_msg: str) -> str:
        """Gemini call for JSON extraction. Returns content string."""
        full_prompt = (
            f"{system_prompt}\n\n---\n\n{user_msg}\n\n"
            "Return ONLY valid JSON, no markdown."
        )
        resp = self.gemini_model.generate_content(
            full_prompt,
            generation_config={
                "temperature": GEMINI_JSON_TEMPERATURE,
                "response_mime_type": "application/json",
            },
        )
        return resp.text

    @RETRY_POLICY
    def _gemini_text(self, prompt: str) -> str:
        """Gemini call for free-text (recommendations). Returns content string."""
        resp = self.gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": GEMINI_RECOMMENDATIONS_TEMPERATURE,
                "max_output_tokens": GEMINI_RECOMMENDATIONS_MAX_TOKENS,
            },
        )
        return resp.text

    # ------------------------------------------------------------------ #
    # THE single shared fallback helper                                  #
    # ------------------------------------------------------------------ #

    def _call_with_fallback(
        self,
        primary: Callable[[], str],
        fallback: Optional[Callable[[], str]],
        *,
        label: str,
        failure_default: str,
    ) -> str:
        """Call ``primary``; on exception, call ``fallback`` if available.

        This method is the SINGLE source of truth for the Groq→Gemini fallback
        pattern.  Every call site goes through it; no caller may re-implement
        the try/except + Gemini-availability check.

        Args:
            primary: zero-arg callable performing the Groq call (already bound
                with its prompt/system-prompt args).
            fallback: zero-arg callable performing the Gemini call, or ``None``
                if Gemini is not available (the public methods pre-compute this
                so the helper does not need to know about ``self.gemini_available``
                — but it does check it defensively).
            label: human-readable label for log lines (e.g. "JSON extraction").
            failure_default: value to return if both providers fail (``""`` for
                JSON callers that will check ``if not content``; a longer string
                for recommendations).

        Returns:
            The content string from whichever provider succeeded, or
            ``failure_default`` if both failed.
        """
        try:
            content = primary()
            logger.info("✅ Groq responded (%s)", label)
            return content
        except Exception as groq_err:
            logger.warning(
                "Groq failed for %s: %s", label, type(groq_err).__name__
            )
            if fallback is not None and self.gemini_available:
                logger.info("🔄 Falling back to Gemini (%s)...", label)
                try:
                    content = fallback()
                    logger.info("✅ Gemini responded (%s)", label)
                    return content
                except Exception as gemini_err:
                    logger.error(
                        " Gemini also failed for %s: %s", label, gemini_err
                    )
                    return failure_default
            logger.error(" No fallback available for %s: %s", label, groq_err)
            return failure_default

    # ------------------------------------------------------------------ #
    # Public API (used by llm_extraction and recommendations)            #
    # ------------------------------------------------------------------ #

    def extract_json(
        self,
        system_prompt: str,
        user_msg: str,
        *,
        label: str = "JSON extraction",
    ) -> str:
        """Groq→Gemini fallback for JSON extraction.

        Returns the raw content string (possibly empty).  The caller is
        responsible for parsing it as JSON.
        """
        fallback = (
            (lambda: self._gemini_json(system_prompt, user_msg))
            if self.gemini_available
            else None
        )
        return self._call_with_fallback(
            primary=lambda: self._groq_json(system_prompt, user_msg),
            fallback=fallback,
            label=label,
            failure_default="",
        )

    def generate_text(
        self,
        prompt: str,
        *,
        label: str = "text generation",
        failure_default: str = "",
    ) -> str:
        """Groq→Gemini fallback for free-text generation.

        Returns the content string, or ``failure_default`` if both providers
        fail.
        """
        fallback = (
            (lambda: self._gemini_text(prompt))
            if self.gemini_available
            else None
        )
        return self._call_with_fallback(
            primary=lambda: self._groq_text(prompt),
            fallback=fallback,
            label=label,
            failure_default=failure_default,
        )
