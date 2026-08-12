"""Local JSON-file cache of car specs.

Replaces the original module-level ``local_cars_db = []`` global list with a
small ``LocalCarsDB`` class.  All callers receive an instance of this class
(constructed once in ``main.py``) and never touch global mutable state.

File format (``local_cars_db.json``) is unchanged, so existing caches load
without migration.
"""
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
    """Wraps the ``local_cars_db.json`` file.

    The in-memory list is private (``self._cars``); callers interact via
    :meth:`find` and :meth:`save`.  No module-level mutable state.
    """

    def __init__(self, path: Path = LOCAL_DB_PATH) -> None:
        self.path: Path = path
        self._cars: List[Dict[str, Any]] = []
        self._load()

    # ---- internal ----------------------------------------------------------

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

    # ---- public API --------------------------------------------------------

    def find(
        self,
        make: str,
        model: str,
        cc: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a car by make+model, optionally matching engine displacement.

        Returns a specs dict built by :func:`utils.build_specs_dict` (single
        source of truth for the shape + defaults), or ``None`` if no match.

        Matching rules (preserved exactly from the original ``find_in_local_db``):
            - make and model are case-insensitive, stripped.
            - If ``cc`` is given, the stored ``engine_displacement_liters``
              must be within ``DISPLACEMENT_MATCH_TOLERANCE_L`` litres.
            - If ``cc`` is omitted, the first make+model match wins.
        """
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
        """Add a car record to the DB and persist to disk.

        No-op if an identical make+model+displacement record already exists.
        The record is built by :func:`utils.build_car_record` (which reuses
        :func:`utils.build_specs_dict`) — no duplicate defaulting logic here.
        """
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
