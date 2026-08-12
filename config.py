"""Centralized configuration: environment variables, named constants, retry policy.

This module is the SINGLE source of truth for every magic number, magic string,
and tunable parameter used anywhere else in the codebase.  No other module
hardcodes paths, model names, temperatures, thresholds, or defaults — they all
import from here.

Importing this module has side effects: it calls ``load_dotenv()`` so that
``GROQ_API_KEY`` / ``TAVILY_API_KEY`` / ``GEMINI_API_KEY`` are populated before
any other module reads them.  This matches the original behaviour where
``load_dotenv()`` ran at import time of the single-file app.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# ---- Environment (loaded once, at import time) -----------------------------
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# ---- File paths (CWD-relative, identical to the original) ------------------
# Kept relative to preserve the exact on-disk layout of the original app.
LOCAL_DB_PATH: Path = Path("local_cars_db.json")
CITY_MODEL_PATH: Path = Path("city_model.pkl")
HIGHWAY_MODEL_PATH: Path = Path("highway_model.pkl")

# ---- Google Drive IDs for the XGBoost .pkl models --------------------------
CITY_MODEL_GDRIVE_ID: str = "1zkL48TzAL2WfkaO7FymEJ49ttgrGm8Tw"
HIGHWAY_MODEL_GDRIVE_ID: str = "1Hb0i83uGj5MWsmpKhiueYalVp-0v-hw6"
GDRIVE_URL_TEMPLATE: str = "https://drive.google.com/uc?id={file_id}"

# ---- LLM model identifiers --------------------------------------------------
GROQ_MODEL: str = "llama-3.1-8b-instant"
GEMINI_MODEL: str = "gemini-2.0-flash"

# ---- LLM generation parameters ---------------------------------------------
GROQ_JSON_TEMPERATURE: float = 0.1
GROQ_RECOMMENDATIONS_TEMPERATURE: float = 0.7
GROQ_RECOMMENDATIONS_MAX_TOKENS: int = 600
GEMINI_JSON_TEMPERATURE: float = 0.1
GEMINI_RECOMMENDATIONS_TEMPERATURE: float = 0.7
GEMINI_RECOMMENDATIONS_MAX_TOKENS: int = 600

# ---- Tavily search parameters ----------------------------------------------
TAVILY_SEARCH_DEPTH: str = "advanced"
TAVILY_MAX_RESULTS: int = 5
MULTI_QUERY_MIN_CONTEXTS: int = 2  # stop after this many non-empty contexts

# ---- Context / log truncation ----------------------------------------------
MAX_CONTEXT_LENGTH: int = 4000
LLM_LOG_PREVIEW_LENGTH: int = 500
JSON_PARSE_ERROR_PREVIEW: int = 300

# ---- Fuel-consumption formula constants (adjust_consumption) ---------------
# Preserved EXACTLY from the original: 235.21 (not the textbook 235.215).
MPG_TO_LPER100KM_FACTOR: float = 235.21
AC_MULTIPLIER_EXTREME_HEAT: float = 1.20   # AC on AND temperature > 35 C
AC_MULTIPLIER_NORMAL: float = 1.08         # AC on AND temperature <= 35 C
BASE_ADJUSTMENT_MULTIPLIER: float = 1.20   # applied unconditionally to every trip
EXTREME_HEAT_THRESHOLD_C: float = 35.0

# ---- Engine-displacement helpers -------------------------------------------
CC_TO_LITERS_THRESHOLD: float = 10.0   # values > 10 are treated as cc, else liters
DISPLACEMENT_MATCH_TOLERANCE_L: float = 0.05  # local-DB match tolerance

# ---- Default car-spec values -----------------------------------------------
# NOTE: the original code used TWO different fuel_type defaults.  We preserve
# that exactly so the API response is byte-for-byte identical.
#   - local DB / build_specs_dict  -> "Regular Gasoline"
#   - apply_safety_net             -> "Gasoline"
# normalize_fuel_type() maps both to "Regular Gasoline" for the ML model, but
# the raw value is exposed in the /predict response, so changing it would
# alter the API contract.
DEFAULT_ENGINE_CYLINDERS: int = 4
DEFAULT_DRIVE: str = "FWD"
DEFAULT_FUEL_TYPE_LOCAL_DB: str = "Regular Gasoline"
DEFAULT_FUEL_TYPE_SAFETY_NET: str = "Gasoline"
DEFAULT_TURBOCHARGER: bool = False

# ---- Vehicle age ------------------------------------------------------------
CURRENT_YEAR: int = 2026

# ---- Shared retry policy (applied to every LLM primitive) ------------------
# Single decorator instance reused by every @RETRY_POLICY-decorated method in
# llm_client.py — replaces four identical @retry(...) lines in the original.
RETRY_POLICY: Any = retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
)
