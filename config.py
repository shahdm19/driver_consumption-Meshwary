from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

LOCAL_DB_PATH: Path = Path("local_cars_db.json")
CITY_MODEL_PATH: Path = Path("city_model.pkl")
HIGHWAY_MODEL_PATH: Path = Path("highway_model.pkl")

CITY_MODEL_GDRIVE_ID: str = "1zkL48TzAL2WfkaO7FymEJ49ttgrGm8Tw"
HIGHWAY_MODEL_GDRIVE_ID: str = "1Hb0i83uGj5MWsmpKhiueYalVp-0v-hw6"
GDRIVE_URL_TEMPLATE: str = "https://drive.google.com/uc?id={file_id}"

GROQ_MODEL: str = "llama-3.1-8b-instant"
GEMINI_MODEL: str = "gemini-2.0-flash"

GROQ_JSON_TEMPERATURE: float = 0.1
GROQ_RECOMMENDATIONS_TEMPERATURE: float = 0.7
GROQ_RECOMMENDATIONS_MAX_TOKENS: int = 600
GEMINI_JSON_TEMPERATURE: float = 0.1
GEMINI_RECOMMENDATIONS_TEMPERATURE: float = 0.7
GEMINI_RECOMMENDATIONS_MAX_TOKENS: int = 600

TAVILY_SEARCH_DEPTH: str = "advanced"
TAVILY_MAX_RESULTS: int = 5
MULTI_QUERY_MIN_CONTEXTS: int = 2  # stop after this many non-empty contexts

MAX_CONTEXT_LENGTH: int = 4000
LLM_LOG_PREVIEW_LENGTH: int = 500
JSON_PARSE_ERROR_PREVIEW: int = 300

MPG_TO_LPER100KM_FACTOR: float = 235.21
AC_MULTIPLIER_EXTREME_HEAT: float = 1.20   # AC on AND temperature > 35 C
AC_MULTIPLIER_NORMAL: float = 1.08         # AC on AND temperature <= 35 C
BASE_ADJUSTMENT_MULTIPLIER: float = 1.20   # applied unconditionally to every trip
EXTREME_HEAT_THRESHOLD_C: float = 35.0

CC_TO_LITERS_THRESHOLD: float = 10.0   # values > 10 are treated as cc, else liters
DISPLACEMENT_MATCH_TOLERANCE_L: float = 0.05  # local-DB match tolerance

DEFAULT_ENGINE_CYLINDERS: int = 4
DEFAULT_DRIVE: str = "FWD"
DEFAULT_FUEL_TYPE_LOCAL_DB: str = "Regular Gasoline"
DEFAULT_FUEL_TYPE_SAFETY_NET: str = "Gasoline"
DEFAULT_TURBOCHARGER: bool = False

CURRENT_YEAR: int = 2026
RETRY_POLICY: Any = retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
)
