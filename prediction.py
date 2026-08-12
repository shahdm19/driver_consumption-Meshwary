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

FUEL_DIESEL_KEYWORDS: List[str] = ["diesel", "سولار", "ديزل"]
FUEL_PREMIUM_KEYWORDS: List[str] = ["95", "premium", "سوبر", "بنزين 95"]
FUEL_MIDGRADE_KEYWORDS: List[str] = ["92", "midgrade", "متوسط", "بنزين 92"]
FUEL_REGULAR_KEYWORDS: List[str] = [
    "80", "regular", "بنزين 80", "gasoline", "petrol", "بنزين",
]

FUEL_TYPE_DIESEL: str = "Diesel"
FUEL_TYPE_PREMIUM: str = "Premium Gasoline"
FUEL_TYPE_MIDGRADE: str = "Midgrade Gasoline"
FUEL_TYPE_REGULAR: str = "Regular Gasoline"


def normalize_fuel_type(fuel: Optional[str]) -> str:
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


def adjust_consumption(mpg: float, temperature: float, ac_on: bool) -> float:

    liters = MPG_TO_LPER100KM_FACTOR / mpg
    if ac_on:
        if temperature > EXTREME_HEAT_THRESHOLD_C:
            liters *= AC_MULTIPLIER_EXTREME_HEAT
        else:
            liters *= AC_MULTIPLIER_NORMAL
    liters *= BASE_ADJUSTMENT_MULTIPLIER
    return round(liters, 2)


def _ensure_model_exists(path, gdrive_id: str) -> None:
    if not os.path.exists(path):
        url = GDRIVE_URL_TEMPLATE.format(file_id=gdrive_id)
        logger.info("Downloading model from %s → %s", url, path)
        gdown.download(url, str(path), quiet=False)


class ConsumptionPredictor:

    def __init__(self) -> None:
        _ensure_model_exists(CITY_MODEL_PATH, CITY_MODEL_GDRIVE_ID)
        _ensure_model_exists(HIGHWAY_MODEL_PATH, HIGHWAY_MODEL_GDRIVE_ID)
        self.city_model = joblib.load(str(CITY_MODEL_PATH))
        self.highway_model = joblib.load(str(HIGHWAY_MODEL_PATH))
        logger.info("ML models loaded (city + highway)")

    def _build_feature_frame(
        self, trip: TripInput, specs: Dict[str, Any]
    ) -> pd.DataFrame:

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
        features = self._build_feature_frame(trip, specs)
        if trip.road_type.lower().strip() == "city":
            mpg = self.city_model.predict(features)[0]
        else:
            mpg = self.highway_model.predict(features)[0]
        return adjust_consumption(mpg, trip.temperature, trip.ac_on)
