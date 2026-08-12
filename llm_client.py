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
    def __init__(self) -> None:
        self.groq = None
        self.gemini_model = None
        self.gemini_available: bool = False

        if GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized")
        else:
            logger.warning("Groq client not initialized (missing key or library)")

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


    def _call_with_fallback(
        self,
        primary: Callable[[], str],
        fallback: Optional[Callable[[], str]],
        *,
        label: str,
        failure_default: str,
    ) -> str:

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

    def extract_json(
        self,
        system_prompt: str,
        user_msg: str,
        *,
        label: str = "JSON extraction",
    ) -> str:
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
