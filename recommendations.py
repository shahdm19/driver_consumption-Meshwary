from __future__ import annotations
from typing import Tuple
from config import DEFAULT_ENGINE_CYLINDERS
from llm_client import LLMClient
from logging_config import get_logger
from models import TripInput
from prompts import RECOMMENDATIONS_PROMPT_TEMPLATE
from utils import calculate_car_age

logger = get_logger(__name__)

LONG_DISTANCE_ROUTES: set = {
    ("cairo", "alexandria"), ("alexandria", "cairo"),
    ("cairo", "hurghada"), ("hurghada", "cairo"),
    ("cairo", "sharm"), ("sharm", "cairo"),
    ("cairo", "luxor"), ("luxor", "cairo"),
    ("cairo", "aswan"), ("aswan", "cairo"),
    ("cairo", "mansoura"), ("mansoura", "cairo"),
    ("cairo", "ismailia"), ("ismailia", "cairo"),
    ("cairo", "suez"), ("suez", "cairo"),
    ("cairo", "tanta"), ("tanta", "cairo"),
    ("cairo", "assiut"), ("assiut", "cairo"),
}

URBAN_ROUTES: set = {
    ("cairo", "giza"), ("giza", "cairo"),
    ("nasr city", "maadi"), ("maadi", "nasr city"),
    ("nasr city", "new cairo"), ("new cairo", "nasr city"),
    ("cairo", "6th october"), ("6th october", "cairo"),
    ("cairo", "new cairo"), ("new cairo", "cairo"),
}

RECOMMENDATIONS_FAILURE_DEFAULT: str = (
    "Unable to generate recommendations at this time. Please try again later."
)


def _age_category(year: int) -> str:
    age = calculate_car_age(year)
    if age <= 2:
        return "new (less than 2 years old)"
    if age <= 5:
        return "relatively new (2-5 years old)"
    if age <= 10:
        return "medium age (5-10 years old)"
    return "older vehicle (more than 10 years old)"


def get_route_context(from_loc: str, to_loc: str, road_type: str) -> str:
    from_loc = from_loc.strip().lower()
    to_loc = to_loc.strip().lower()
    route_tuple: Tuple[str, str] = (from_loc, to_loc)

    if route_tuple in LONG_DISTANCE_ROUTES:
        return (
            "- This is a LONG-DISTANCE intercity trip in Egypt (~200+ km)\n"
            "- Road is typically a desert highway with: open road, fewer stops, "
            "higher speeds (90-120 km/h)\n"
            "- Possible challenges: crosswinds, sand on road, limited fuel "
            "stations, fatigue\n"
            f"- Weather consideration: {road_type} driving with potential "
            "temperature variations\n"
            "- Best fuel-saving strategy: maintain steady highway speed, use "
            "cruise control if available"
        )

    if route_tuple in URBAN_ROUTES:
        return (
            "- This is an URBAN trip within Egyptian city traffic\n"
            "- Road has: heavy traffic, frequent stops, traffic lights, low "
            "speeds (10-40 km/h)\n"
            "- Possible challenges: stop-and-go traffic, idling, frequent "
            "braking\n"
            "- AC usage has higher impact in city driving due to idling\n"
            "- Best fuel-saving strategy: avoid aggressive acceleration, "
            "anticipate stops, minimize idling"
        )

    if from_loc == to_loc:
        return (
            f"- This appears to be a LOCAL trip within {from_loc.title()}\n"
            "- Likely city driving with frequent stops and traffic\n"
            "- Short distance, but fuel efficiency still matters for daily "
            "commutes"
        )

    if road_type.lower() == "city":
        return (
            f"- Trip from {from_loc.title()} to {to_loc.title()}\n"
            "- City driving conditions: traffic, stops, lower speeds\n"
            "- Focus on urban fuel-saving techniques"
        )

    return (
        f"- Trip from {from_loc.title()} to {to_loc.title()}\n"
        "- Highway driving conditions: open road, higher speeds\n"
        "- Focus on highway fuel-saving techniques (steady speed, aerodynamics)"
    )


def get_recommendations(
    trip: TripInput,
    consumption: float,
    specs: dict,
    llm_client: LLMClient,
) -> str:
    route_context = get_route_context(
        trip.from_location, trip.to_location, trip.road_type
    )
    prompt = RECOMMENDATIONS_PROMPT_TEMPLATE.format(
        make=trip.make,
        model=trip.model,
        year=trip.year,
        age_category=_age_category(trip.year),
        engine_displacement_liters=specs.get("engine_displacement_liters"),
        engine_cylinders=specs.get("engine_cylinders", DEFAULT_ENGINE_CYLINDERS),
        turbocharger="Yes" if specs.get("turbocharger") else "No",
        fuel_type=specs.get("fuel_type"),
        road_type=trip.road_type,
        temperature=trip.temperature,
        ac_on="On" if trip.ac_on else "Off",
        consumption=consumption,
        from_location=trip.from_location,
        to_location=trip.to_location,
        route_context=route_context,
    )

    return llm_client.generate_text(
        prompt,
        label="recommendations",
        failure_default=RECOMMENDATIONS_FAILURE_DEFAULT,
    )
