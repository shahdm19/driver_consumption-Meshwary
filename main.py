from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import FastAPI
from config import GEMINI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY
from llm_client import LLMClient
from llm_extraction import apply_safety_net, get_car_specs
from local_db import LocalCarsDB
from logging_config import get_logger, setup_logging
from models import TripInput
from prediction import ConsumptionPredictor
from recommendations import get_recommendations
from search import WebSearcher

setup_logging()
logger = get_logger(__name__)
local_db = LocalCarsDB()
searcher = WebSearcher(api_key=TAVILY_API_KEY)
llm_client = LLMClient()
predictor = ConsumptionPredictor()

app = FastAPI()

@app.post("/predict")
def predict(trip: TripInput) -> Dict[str, Any]:

    logger.info(
        "📡 Predict request: %s %s %s, cc=%s",
        trip.make, trip.model, trip.year, trip.cc,
    )

    specs = get_car_specs(
        trip.make, trip.model, trip.year, trip.cc,
        local_db, searcher, llm_client,
    )
    if not specs:
        return {
            "status": "error",
            "message": "Failed to fetch car data. Please try again.",
        }

    safe_specs = apply_safety_net(specs)
    if safe_specs is None:
        return {
            "status": "missing_critical_data",
            "message": (
                f"تم العثور على سيارة {trip.make} {trip.model}، لكن لم نتمكن "
                "من تأكيد سعة المحرك بدقة."
            ),
            "missing_fields": ["engine_displacement_liters"],
            "suggested_options": [1.4, 1.6, 2.0],
        }

    consumption = predictor.predict(trip, safe_specs)
    logger.info("Consumption calculated: %s L/100km", consumption)

    recommendations = get_recommendations(trip, consumption, safe_specs, llm_client)
    logger.info("Recommendations generated")

    return {
        "status": "success",
        "consumption_rate": consumption,
        "recommendations": recommendations,
        "specs_used": {
            "engine_displacement_liters": safe_specs.get("engine_displacement_liters"),
            "engine_cylinders": safe_specs.get("engine_cylinders"),
            "turbocharger": safe_specs.get("turbocharger"),
            "drive": safe_specs.get("drive"),
            "fuel_type": safe_specs.get("fuel_type"),
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "tavily_available": searcher.available,
        "gemini_available": llm_client.gemini_available,
    }


@app.get("/ping")
def ping() -> Dict[str, bool]:
    return {"pong": True}


@app.get("/debug/env")
def debug_env() -> Dict[str, Any]:
    return {
        "groq_key_set": bool(GROQ_API_KEY),
        "tavily_key_set": bool(TAVILY_API_KEY),
        "groq_key_prefix": (GROQ_API_KEY or "")[:6] + "...",
        "tavily_key_prefix": (TAVILY_API_KEY or "")[:6] + "...",
        "cwd": os.getcwd(),
        "env_file_exists": os.path.exists(".env"),
    }


@app.get("/debug/models")
def debug_models() -> Dict[str, Any]:
    return {
        "city_model_loaded": predictor.city_model is not None,
        "highway_model_loaded": predictor.highway_model is not None,
        "city_model_type": str(type(predictor.city_model).__name__),
        "highway_model_type": str(type(predictor.highway_model).__name__),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
