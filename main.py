from fastapi import FastAPI
import joblib
import pandas as pd
import gdown
import os
import json
import logging
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS
from dotenv import load_dotenv


if not os.path.exists('city_model.pkl'):
    gdown.download('https://drive.google.com/uc?id=1zkL48TzAL2WfkaO7FymEJ49ttgrGm8Tw', 'city_model.pkl', quiet=False)

if not os.path.exists('highway_model.pkl'):
    gdown.download('https://drive.google.com/uc?id=1Hb0i83uGj5MWsmpKhiueYalVp-0v-hw6', 'highway_model.pkl', quiet=False)

app = FastAPI()

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

city_model = joblib.load('city_model.pkl')
highway_model = joblib.load('highway_model.pkl')


MAIN_SYSTEM_PROMPT = """
You are an expert automotive data extraction assistant specializing in the Egyptian and Middle Eastern automotive market. 

Your task is to extract the technical specifications of a vehicle from the provided search results based on the user's input (Make, Model, and Year).

CRITICAL RULES:
1. MARKET SPECIFICITY: You MUST prioritize specifications for cars sold in Egypt and the Middle East. DO NOT default to US or European specifications.
2. BEST-SELLING DEFAULT: If multiple variants exist, you MUST default to the most common or best-selling variant in the Egyptian market.
3. NO HALLUCINATION: If a specific detail cannot be found in the search results, you MUST return `null` for that key. DO NOT guess.
4. STRICT OUTPUT: You MUST return the output ONLY as a valid, raw JSON object. Do not include any text or markdown formatting outside the JSON object.

REQUIRED JSON SCHEMA:
{
  "make": "string",
  "model": "string",
  "year": integer,
  "engine_displacement_liters": float,
  "engine_cylinders": integer,
  "drive": "string (Must be exactly one of: 'FWD', 'RWD', 'AWD', '4WD')",
  "fuel_type": "string (e.g., 'Gasoline', 'Diesel', 'Hybrid')",
  "turbocharger": boolean
}
"""

DISP_PROMPT = """
You are an expert in the Egyptian automotive market. 
Extract ONLY the engine displacement in liters for this specific car from the search results.
CRITICAL RULES: Prioritize Egyptian market specs. NO hallucinations. Use null if not found.
Respond ONLY with valid JSON: {"engine_displacement_liters": <float or null>}
Look for keywords like "سعة المحرك", "لتر", "cc" (if cc, divide by 1000).
"""

TURBO_PROMPT = """
You are an expert in the Egyptian automotive market.
Determine ONLY if this car has a Turbocharger from the search results.
CRITICAL RULES: Prioritize Egyptian market specs. NO hallucinations. Use null if not found.
Respond ONLY with valid JSON: {"turbocharger": <boolean or null>}
Look for keywords like "تيربو", "Turbo", "T-GDI", "شاحن توربيني".
"""


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([r['body'] for r in results])
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return ""

def extract_specs_with_llm(context: str, system_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Search Results:\n{context}\n\nExtract the data."}
            ],
            temperature=0.1,
            response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        return {}

def get_car_specs(make: str, model: str, year: int) -> dict:
   
    main_query = f"مواصفات سيارة {make} {model} موديل {year} في السوق المصري مواصفات فنية"
    context = search_web(main_query)
    
    specs = extract_specs_with_llm(context, MAIN_SYSTEM_PROMPT)
    if not specs:
        return None

   
    if specs.get("engine_displacement_liters") is None:
        logging.warning("Engine displacement is null. Triggering DEDICATED search...")
        disp_query = f"سعة محرك سيارة {make} {model} {year} باللتر أو سي سي في مصر"
        disp_context = search_web(disp_query)
        disp_result = extract_specs_with_llm(disp_context, DISP_PROMPT)
        if disp_result and disp_result.get("engine_displacement_liters") is not None:
            specs["engine_displacement_liters"] = disp_result["engine_displacement_liters"]

    if specs.get("turbocharger") is None:
        logging.warning("Turbocharger status is null. Triggering DEDICATED search...")
        turbo_query = f"هل محرك سيارة {make} {model} {year} يحتوي على شاحن توربيني Turbo في مصر"
        turbo_context = search_web(turbo_query)
        turbo_result = extract_specs_with_llm(turbo_context, TURBO_PROMPT)
        if turbo_result and "turbocharger" in turbo_result:
            specs["turbocharger"] = turbo_result["turbocharger"]

    return specs


class TripInput(BaseModel):
    make: str
    model: str
    year: int  
    road_type: str
    temperature: float
    ac_on: bool
    from_location: str = ""
    to_location: str = ""

def adjust_consumption(mpg, temperature, ac_on):
    liters = 235.21 / mpg
    if ac_on:
        if temperature > 35:
            liters *= 1.20
        else:
            liters *= 1.08
    liters *= 1.20
    return round(liters, 2)

def predict_consumption(trip: TripInput) -> float:
    specs = get_car_specs(trip.make, trip.model, trip.year)
    
   
    if not specs or specs.get("engine_displacement_liters") is None:
        return None

    drive = specs.get("drive", "FWD")
    drive_4wd = 1 if drive == '4WD' else 0
    drive_rwd = 1 if drive == 'RWD' else 0

    fuel = specs.get("fuel_type", "Gasoline")
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    fuel_midgrade = 1 if fuel == 'Midgrade Gasoline' else 0
    fuel_premium = 1 if fuel == 'Premium Gasoline' else 0

   
    input_data = pd.DataFrame({
        'Engine Displacement': [specs["engine_displacement_liters"]],
        'Engine Cylinders': [specs.get("engine_cylinders", 4)], 
        'Turbocharger': [1 if specs.get("turbocharger") else 0], 
        'Year': [trip.year],
        'Drive_4WD': [drive_4wd],
        'Drive_RWD': [drive_rwd],
        'Fuel Type 1_Diesel': [fuel_diesel],
        'Fuel Type 1_Midgrade Gasoline': [fuel_midgrade],
        'Fuel Type 1_Premium Gasoline': [fuel_premium],
    })

    if trip.road_type == 'city':
        mpg = city_model.predict(input_data)[0]
    else:
        mpg = highway_model.predict(input_data)[0]

    return adjust_consumption(mpg, trip.temperature, trip.ac_on)

def get_recommendations(trip: TripInput, consumption: float, specs: dict) -> str:

    car_age = 2026 - trip.year
    
    if car_age <= 2:
        age_category = "new (less than 2 years old)"
    elif car_age <= 5:
        age_category = "relatively new (2-5 years old)"
    elif car_age <= 10:
        age_category = "medium age (5-10 years old)"
    else:
        age_category = "older vehicle (more than 10 years old)"
    
    prompt = f"""
You are an expert automotive advisor with deep knowledge of fuel efficiency and road safety.

Trip Details:
- Car: {trip.make} {trip.model} (Model Year: {trip.year})
- Vehicle Age Category: {age_category}
- Road Type: {trip.road_type}
- Temperature: {trip.temperature}°C
- AC: {"On" if trip.ac_on else "Off"}
- Fuel Consumption Rate: {consumption} L/100km
- Route: {trip.from_location} to {trip.to_location}

Based on the specific conditions above, provide exactly 5 professional and personalized recommendations that:
1. Are directly related to the given car model, road type, and weather conditions
2. Take into account the vehicle's age category for maintenance-related tips
3. Focus on reducing fuel consumption and improving efficiency
4. Include important safety precautions relevant to these conditions

Be specific, practical, and professional. Avoid generic advice.
Format each tip as a clear actionable point.
"""
    
    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.choices[0].message.content

@app.post("/predict")
def predict(trip: TripInput):
    consumption = predict_consumption(trip)
    
    if consumption is None:
        return {
            "error": "Unable to determine fuel consumption due to missing critical specifications"
        }
    recommendations = get_recommendations(trip, consumption)
    
    return {
        "consumption_rate": consumption,
        "recommendations": recommendations
    }