from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DISPLACEMENT_MATCH_TOLERANCE_L, LOCAL_DB_PATH
from logging_config import get_logger
from utils import build_car_record, build_specs_dict, cc_to_liters

logger = get_logger(__name__)


class LocalCarsDB:
    def __init__(self, path: Path = LOCAL_DB_PATH) -> None:
        self.path: Path = path
        self._cars: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load from disk, or start empty if the file is missing/corrupt."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._cars = json.load(f)
                logger.info(" Loaded %d cars from local DB", len(self._cars))
            except Exception as e:
                logger.error(" Failed to load local DB: %s", e)
                self._cars = []
        else:
            logger.warning("%s not found. Starting with empty DB.", self.path)

    def find(
        self,
        make: str,
        model: str,
        cc: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:

        make_lower = make.lower().strip()
        model_lower = model.lower().strip()
        logger.info("Searching local DB for: %s %s (cc=%s)", make, model, cc)

        cc_in_liters: Optional[float] = cc_to_liters(cc) if cc is not None else None

        for car in self._cars:
            car_make = car.get("make", "").lower().strip()
            car_model = car.get("model", "").lower().strip()
            if car_make == make_lower and car_model == model_lower:
                if cc_in_liters is not None:
                    car_cc = car.get("engine_displacement_liters", 0)
                    if abs(car_cc - cc_in_liters) < DISPLACEMENT_MATCH_TOLERANCE_L:
                        logger.info(
                            "Found EXACT match in Local DB: %s %s %sL",
                            make, model, cc_in_liters,
                        )
                        return build_specs_dict(car)
                else:
                    return build_specs_dict(car)

        logger.info("Not found in Local DB: %s %s", make, model)
        return None

    def save(self, make: str, model: str, year: int, specs: Dict[str, Any]) -> None:
        new_car = build_car_record(make, model, specs)

        for car in self._cars:
            if (
                car["make"].lower().strip() == make.lower().strip()
                and car["model"].lower().strip() == model.lower().strip()
                and car["engine_displacement_liters"]
                == new_car["engine_displacement_liters"]
            ):
                logger.info("Car already exists in Local DB: %s %s", make, model)
                return

        self._cars.append(new_car)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._cars, f, ensure_ascii=False, indent=4)
            logger.info("✅ Saved %s %s to local DB file.", make, model)
        except Exception as e:
            logger.error("❌ Failed to write to local DB file: %s", e)
