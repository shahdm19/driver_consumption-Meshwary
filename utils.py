"""Shared pure-function utilities.

This module exists purely to host helpers that are needed by MORE than one
other module, so that each helper has exactly ONE definition.  No other
module may re-implement cc→liters conversion, specs-dict construction, or
car-age calculation — they must import from here.
"""
from __future__ import annotations

from typing import Any, Dict

from config import (
    CC_TO_LITERS_THRESHOLD,
    CURRENT_YEAR,
    DEFAULT_DRIVE,
    DEFAULT_ENGINE_CYLINDERS,
    DEFAULT_FUEL_TYPE_LOCAL_DB,
    DEFAULT_TURBOCHARGER,
)


def cc_to_liters(cc: float) -> float:
    """Convert a displacement value to liters.

    SINGLE source of truth for the "cc > 10 means cc, else liters" rule.
    Used by:
        - local_db.LocalCarsDB.find        (when matching against stored cars)
        - llm_extraction.get_car_specs     (when overriding with user-provided cc)

    Args:
        cc: Engine displacement.  Values > 10 are treated as cubic centimetres
            (e.g. 1600 → 1.6); values <= 10 are treated as already-in-litres
            (e.g. 1.6 → 1.6).

    Returns:
        Displacement in litres, rounded to 2 decimals.
    """
    if cc > CC_TO_LITERS_THRESHOLD:
        return round(cc / 1000.0, 2)
    return round(cc, 2)


def build_specs_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build a 5-field specs dict with consistent defaults.

    SINGLE source of truth for the spec-field shape and the local-DB defaults.
    Used by:
        - local_db.LocalCarsDB.find    (returns this shape to callers)
        - local_db.LocalCarsDB.save    (via build_car_record, for storage)
        - any future caller that needs the canonical specs shape

    The defaults match the original ``find_in_local_db`` / ``save_to_local_db``
    ``.get(key, default)`` calls exactly.
    """
    return {
        "engine_displacement_liters": raw.get("engine_displacement_liters"),
        "engine_cylinders": raw.get("engine_cylinders", DEFAULT_ENGINE_CYLINDERS),
        "drive": raw.get("drive", DEFAULT_DRIVE),
        "fuel_type": raw.get("fuel_type", DEFAULT_FUEL_TYPE_LOCAL_DB),
        "turbocharger": raw.get("turbocharger", DEFAULT_TURBOCHARGER),
    }


def build_car_record(make: str, model: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build a full car record (with make/model) for storage in the local DB.

    Reuses :func:`build_specs_dict` for the spec fields — does NOT re-implement
    the defaulting logic.  Used by ``local_db.LocalCarsDB.save``.
    """
    return {
        "make": make.strip(),
        "model": model.strip(),
        **build_specs_dict(raw),
    }


def calculate_car_age(year: int) -> int:
    """SINGLE source of truth for car-age calculation.

    Used by:
        - prediction.ConsumptionPredictor._build_feature_frame
        - recommendations._age_category

    Both used to hardcode ``2026 - year``; now they share this function and
    the ``CURRENT_YEAR`` constant from config.
    """
    return CURRENT_YEAR - year
