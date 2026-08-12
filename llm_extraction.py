"""High-level spec extraction: local DB → web search → LLM → safety net.

This module orchestrates the pipeline.  It does NOT contain any fallback
logic — that lives in :mod:`llm_client`.  It does NOT contain any prompt
strings — those live in :mod:`prompts`.  It does NOT contain any cc→liters
or specs-dict logic — those live in :mod:`utils`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from config import (
    DEFAULT_DRIVE,
    DEFAULT_ENGINE_CYLINDERS,
    DEFAULT_FUEL_TYPE_SAFETY_NET,
    DEFAULT_TURBOCHARGER,
    JSON_PARSE_ERROR_PREVIEW,
    LLM_LOG_PREVIEW_LENGTH,
    MAX_CONTEXT_LENGTH,
)
from llm_client import LLMClient
from local_db import LocalCarsDB
from logging_config import get_logger
from prompts import DISP_PROMPT, MAIN_SYSTEM_PROMPT, SPEC_EXTRACTION_USER_MSG_TEMPLATE
from search import WebSearcher
from utils import cc_to_liters

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# JSON parsing — single helper for the "parse + log on failure" pattern        #
# --------------------------------------------------------------------------- #

def _parse_json_response(content: str, label: str) -> Dict[str, Any]:
    """Parse an LLM JSON response with consistent error logging.

    SINGLE source of truth for the JSON-parse + log pattern.  Returns ``{}``
    on any parse failure (preserving the original behaviour where
    ``extract_specs_with_llm`` returned ``{}`` on JSONDecodeError).
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        preview = content[:JSON_PARSE_ERROR_PREVIEW] if content else "N/A"
        logger.error("JSON parse failed for %s: %s\nRaw: %s", label, e, preview)
        return {}
    except Exception as e:
        logger.error("Unexpected parse error for %s: %s", label, e)
        return {}


# --------------------------------------------------------------------------- #
# Single-spec extraction via LLM                                              #
# --------------------------------------------------------------------------- #

def extract_specs_with_llm(
    make: str,
    model: str,
    year: int,
    context: str,
    system_prompt: str,
    llm_client: LLMClient,
    expect_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Send ``context`` to the LLM (Groq→Gemini) and parse JSON specs.

    Uses :meth:`LLMClient.extract_json` — the SINGLE source of truth for the
    fallback pattern — and :func:`_parse_json_response` — the SINGLE source of
    truth for JSON parsing.

    Args:
        make/model/year: car identity.
        context: search-results text.
        system_prompt: which system prompt to use (MAIN / DISP / TURBO).
        llm_client: shared :class:`LLMClient` instance.
        expect_field: label for logging (e.g. ``"all specs"``,
            ``"engine_displacement_liters"``).

    Returns:
        Parsed dict, or ``{}`` on any failure.
    """
    if not context:
        logger.warning("No context provided to LLM (field=%s)", expect_field)
        return {}

    if len(context) > MAX_CONTEXT_LENGTH:
        logger.info(
            "Truncating context from %d to %d chars",
            len(context), MAX_CONTEXT_LENGTH,
        )
        context = context[:MAX_CONTEXT_LENGTH]

    user_msg = SPEC_EXTRACTION_USER_MSG_TEMPLATE.format(
        make=make, model=model, year=year, context=context
    )

    label = expect_field or "all specs"
    logger.info(" Sending to LLM (expecting: %s)...", label)

    content = llm_client.extract_json(
        system_prompt,
        user_msg,
        label=f"JSON extraction ({label})",
    )
    if not content:
        logger.error("No content from either LLM")
        return {}

    logger.info("LLM response: %s", content[:LLM_LOG_PREVIEW_LENGTH])
    return _parse_json_response(content, label=label)


# --------------------------------------------------------------------------- #
# Full pipeline: local DB → search → LLM → cache                              #
# --------------------------------------------------------------------------- #

def get_car_specs(
    make: str,
    model: str,
    year: int,
    cc: Optional[float],
    local_db: LocalCarsDB,
    searcher: WebSearcher,
    llm_client: LLMClient,
) -> Optional[Dict[str, Any]]:
    """Resolve car specs via: local DB → web search → LLM extraction.

    Returns the specs dict, or ``None`` if specs cannot be determined.

    The cc→liters override uses :func:`utils.cc_to_liters` (single source of
    truth) — the original code inlined this conversion in THREE places.
    """
    logger.info(" Getting specs for %s %s %s, cc=%s", make, model, year, cc)

    # 1. Try the local DB first (no API calls).
    local_specs = local_db.find(make, model, cc)
    if local_specs:
        logger.info("Using Local DB (no API calls needed)")
        if cc is not None:
            local_specs["engine_displacement_liters"] = cc_to_liters(cc)
        return local_specs

    # 2. Search the web.
    logger.info("️ Not in Local DB. Searching via LLM...")
    context = searcher.multi_query_search(make, model, year)
    logger.info("Combined context length: %d", len(context))

    if not context:
        logger.error(">>> ROOT CAUSE: All searches returned empty")
        context = searcher.search(f"{make} {model} {year} specifications")
    if not context:
        logger.error(">>> Still empty. Cannot proceed without search context.")
        return None

    # 3. Extract via LLM (Groq→Gemini fallback handled inside extract_json).
    specs = extract_specs_with_llm(
        make, model, year, context, MAIN_SYSTEM_PROMPT, llm_client,
        expect_field="all specs",
    )
    if not specs:
        logger.error("Failed to extract specs from main search")
        return None
    logger.info("Initial specs from LLM: %s", specs)

    # 4. Override / fill engine displacement (single cc_to_liters call site).
    if cc is not None:
        specs["engine_displacement_liters"] = cc_to_liters(cc)
        logger.info(
            "Using user-provided cc: %sL", specs["engine_displacement_liters"]
        )
    elif specs.get("engine_displacement_liters") is None:
        logger.warning("Engine displacement is null. Triggering DEDICATED search...")
        disp_query = (
            f'"{make} {model}" {year} engine size displacement liters cc Egypt'
        )
        disp_context = searcher.search(disp_query)
        disp_result = extract_specs_with_llm(
            make, model, year, disp_context, DISP_PROMPT, llm_client,
            expect_field="engine_displacement_liters",
        )
        if disp_result and disp_result.get("engine_displacement_liters") is not None:
            specs["engine_displacement_liters"] = disp_result["engine_displacement_liters"]
            logger.info(
                "Found engine displacement: %sL",
                specs["engine_displacement_liters"],
            )

    # 5. Cache if we have a displacement.
    if specs.get("engine_displacement_liters") is not None:
        local_db.save(make, model, year, specs)

    return specs


# --------------------------------------------------------------------------- #
# Safety net                                                                  #
# --------------------------------------------------------------------------- #

def apply_safety_net(specs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fill ``None`` gaps in LLM-extracted specs with safe defaults.

    Returns ``None`` if ``engine_displacement_liters`` is ``None`` (critical
    missing data) — this triggers the ``missing_critical_data`` API response.

    NOTE on the fuel_type default
    -----------------------------
    The original ``apply_safety_net`` used ``"Gasoline"`` here, while
    ``find_in_local_db`` / ``save_to_local_db`` used ``"Regular Gasoline"``.
    We preserve this exactly via two separate config constants
    (``DEFAULT_FUEL_TYPE_SAFETY_NET`` vs ``DEFAULT_FUEL_TYPE_LOCAL_DB``).
    The downstream :func:`prediction.normalize_fuel_type` maps both to
    ``"Regular Gasoline"`` for the ML model, BUT the raw value is exposed in
    the ``/predict`` response (``specs_used.fuel_type``), so changing it would
    alter the API contract.
    """
    logger.info("🛡️ Applying Safety Net...")

    if specs.get("engine_displacement_liters") is None:
        logger.error(">>> engine_displacement_liters is None after all attempts")
        return None

    if specs.get("drive") is None:
        specs["drive"] = DEFAULT_DRIVE
        logger.info("Set drive to default: %s", DEFAULT_DRIVE)
    if specs.get("turbocharger") is None:
        specs["turbocharger"] = DEFAULT_TURBOCHARGER
        logger.info("Set turbocharger to default: %s", DEFAULT_TURBOCHARGER)
    if specs.get("engine_cylinders") is None:
        specs["engine_cylinders"] = DEFAULT_ENGINE_CYLINDERS
        logger.info("Set engine_cylinders to default: %s", DEFAULT_ENGINE_CYLINDERS)
    if specs.get("fuel_type") is None:
        specs["fuel_type"] = DEFAULT_FUEL_TYPE_SAFETY_NET
        logger.info("Set fuel_type to default: %s", DEFAULT_FUEL_TYPE_SAFETY_NET)

    return specs
