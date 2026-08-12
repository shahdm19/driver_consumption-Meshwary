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
    if cc > CC_TO_LITERS_THRESHOLD:
        return round(cc / 1000.0, 2)
    return round(cc, 2)


def build_specs_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "engine_displacement_liters": raw.get("engine_displacement_liters"),
        "engine_cylinders": raw.get("engine_cylinders", DEFAULT_ENGINE_CYLINDERS),
        "drive": raw.get("drive", DEFAULT_DRIVE),
        "fuel_type": raw.get("fuel_type", DEFAULT_FUEL_TYPE_LOCAL_DB),
        "turbocharger": raw.get("turbocharger", DEFAULT_TURBOCHARGER),
    }


def build_car_record(make: str, model: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "make": make.strip(),
        "model": model.strip(),
        **build_specs_dict(raw),
    }


def calculate_car_age(year: int) -> int:
    return CURRENT_YEAR - year
