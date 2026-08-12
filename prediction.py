"""ML model loading and fuel-consumption prediction.

Owns:
    - XGBoost .pkl download + loading (``ConsumptionPredictor``).
    - Fuel-type normalization (``normalize_fuel_type``) — SINGLE source of truth.
    - The MPG→L/100km + AC + base-adjustment formula (``adjust_consumption``).
    - Feature-frame construction for the XGBoost models.

Car-age calculation is delegated to :func:`utils.calculate_car_age` (shared
with :mod:`recommendations`).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import gdown
import joblib
import pandas as pd

from config import (
    AC_MULTIPLIER_EXTREME_HEAT,
    AC_MULTIPLIER_NORMAL,
    BASE_ADJUSTMENT_MULTIPLIER,
    CITY_MODEL_GDRIVE_ID,
    CITY_MODEL_PATH,
    EXTREME_HEAT_THRESHOLD_C,
    GDRIVE_URL_TEMPLATE,
    HIGHWAY_MODEL_GDRIVE_ID,
    HIGHWAY_MODEL_PATH,
    MPG_TO_LPER100KM_FACTOR,
)
from logging_config import get_logger
from models import TripInput
from utils import calculate_car_age

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Fuel-type normalization — single source of truth                            #
# --------------------------------------------------------------------------- #

# Keyword groups (English + Arabic).  Order matters: the first matching group
# wins, mirroring the original if/elif chain in ``_normalize_fuel_type``.
FUEL_DIESEL_KEYWORDS: List[str] = ["diesel", "سولار", "ديزل"]
FUEL_PREMIUM_KEYWORDS: List[str] = ["95", "premium", "سوبر", "بنزين 95"]
FUEL_MIDGRADE_KEYWORDS: List[str] = ["92", "midgrade", "متوسط", "بنزين 92"]
FUEL_REGULAR_KEYWORDS: List[str] = [
    "80", "regular", "بنزين 80", "gasoline", "petrol", "بنزين",
]

# Canonical fuel-type values accepted by the ML model.
FUEL_TYPE_DIESEL: str = "Diesel"
FUEL_TYPE_PREMIUM: str = "Premium Gasoline"
FUEL_TYPE_MIDGRADE: str = "Midgrade Gasoline"
FUEL_TYPE_REGULAR: str = "Regular Gasoline"


def normalize_fuel_type(fuel: Optional[str]) -> str:
    """Map any fuel string (EN/AR) to one of the 4 canonical ML-model values.

    SINGLE source of truth for fuel-type normalization.  Used only by
    :meth:`ConsumptionPredictor._build_feature_frame` (no other caller), but
    extracted to a named function with named keyword constants so the mapping
    is auditable alongside the prompt's fuel-type instructions in
    :mod:`prompts`.

    Order of checks (preserved exactly from the original):
        diesel → premium → midgrade → regular → default (Regular Gasoline).
    """
    if not fuel:
        return FUEL_TYPE_REGULAR

    fuel_lower = fuel.lower().strip()
    if any(kw in fuel_lower for kw in FUEL_DIESEL_KEYWORDS):
        return FUEL_TYPE_DIESEL
    if any(kw in fuel_lower for kw in FUEL_PREMIUM_KEYWORDS):
        return FUEL_TYPE_PREMIUM
    if any(kw in fuel_lower for kw in FUEL_MIDGRADE_KEYWORDS):
        return FUEL_TYPE_MIDGRADE
    if any(kw in fuel_lower for kw in FUEL_REGULAR_KEYWORDS):
        return FUEL_TYPE_REGULAR

    logger.warning("Unknown fuel_type '%s', defaulting to %s", fuel, FUEL_TYPE_REGULAR)
    return FUEL_TYPE_REGULAR


# --------------------------------------------------------------------------- #
# Consumption formula                                                          #
# --------------------------------------------------------------------------- #

def adjust_consumption(mpg: float, temperature: float, ac_on: bool) -> float:
    """Convert MPG → L/100km and apply AC + base-adjustment multipliers.

    Formula (preserved EXACTLY from the original)::

        liters = 235.21 / mpg
        if ac_on:
            liters *= 1.20 if temperature > 35 else 1.08
        liters *= 1.20   # unconditional base adjustment

    All magic numbers are named constants in :mod:`config`.
    """
    liters = MPG_TO_LPER100KM_FACTOR / mpg
    if ac_on:
        if temperature > EXTREME_HEAT_THRESHOLD_C:
            liters *= AC_MULTIPLIER_EXTREME_HEAT
        else:
            liters *= AC_MULTIPLIER_NORMAL
    liters *= BASE_ADJUSTMENT_MULTIPLIER
    return round(liters, 2)


# --------------------------------------------------------------------------- #
# Model loading + prediction                                                   #
# --------------------------------------------------------------------------- #

def _ensure_model_exists(path, gdrive_id: str) -> None:
    """Download a model from Google Drive if not present locally.

    Replaces the two near-identical ``if not os.path.exists(...): gdown.download(...)``
    blocks at the top of the original file.
    """
    if not os.path.exists(path):
        url = GDRIVE_URL_TEMPLATE.format(file_id=gdrive_id)
        logger.info("Downloading model from %s → %s", url, path)
        gdown.download(url, str(path), quiet=False)


class ConsumptionPredictor:
    """Loads the XGBoost city/highway models and predicts L/100km consumption."""

    def __init__(self) -> None:
        _ensure_model_exists(CITY_MODEL_PATH, CITY_MODEL_GDRIVE_ID)
        _ensure_model_exists(HIGHWAY_MODEL_PATH, HIGHWAY_MODEL_GDRIVE_ID)
        self.city_model = joblib.load(str(CITY_MODEL_PATH))
        self.highway_model = joblib.load(str(HIGHWAY_MODEL_PATH))
        logger.info("ML models loaded (city + highway)")

    def _build_feature_frame(
        self, trip: TripInput, specs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Build the exact feature DataFrame the XGBoost models expect.

        Car-age is computed via :func:`utils.calculate_car_age` (single source
        of truth, shared with :mod:`recommendations`).
        """
        drive = specs.get("drive", "FWD")
        drive_4wd = 1 if drive == "4WD" else 0
        drive_rwd = 1 if drive == "RWD" else 0

        fuel = normalize_fuel_type(specs.get("fuel_type", "Gasoline"))
        logger.info("Normalized fuel_type: %s", fuel)

        fuel_diesel = 1 if fuel == FUEL_TYPE_DIESEL else 0
        fuel_midgrade = 1 if fuel == FUEL_TYPE_MIDGRADE else 0
        fuel_premium = 1 if fuel == FUEL_TYPE_PREMIUM else 0

        car_age = calculate_car_age(trip.year)
        logger.info("Car year: %s → car_age: %s", trip.year, car_age)

        return pd.DataFrame({
            "Engine Displacement": [specs["engine_displacement_liters"]],
            "Engine Cylinders": [specs.get("engine_cylinders", 4)],
            "Turbocharger": [1 if specs.get("turbocharger") else 0],
            "car_age": [car_age],
            "Drive_4WD": [drive_4wd],
            "Drive_RWD": [drive_rwd],
            "Fuel Type 1_Diesel": [fuel_diesel],
            "Fuel Type 1_Midgrade Gasoline": [fuel_midgrade],
            "Fuel Type 1_Premium Gasoline": [fuel_premium],
        })

    def predict(self, trip: TripInput, specs: Dict[str, Any]) -> float:
        """Predict fuel consumption in L/100km for the trip.

        Selects the city or highway model based on ``trip.road_type`` and
        applies :func:`adjust_consumption` to the raw MPG output.
        """
        features = self._build_feature_frame(trip, specs)
        if trip.road_type.lower().strip() == "city":
            mpg = self.city_model.predict(features)[0]
        else:
            mpg = self.highway_model.predict(features)[0]
        return adjust_consumption(mpg, trip.temperature, trip.ac_on)
