from fastapi import FastAPI
import joblib
import pandas as pd
import gdown
import os
import json
import logging
import traceback
from typing import Optional
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


LOCAL_DB_PATH = 'local_cars_db.json'
local_cars_db = []


try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

if not os.path.exists('city_model.pkl'):
    gdown.download(
        'https://drive.google.com/uc?id=1zkL48TzAL2WfkaO7FymEJ49ttgrGm8Tw',
        'city_model.pkl',
        quiet=False
    )

if not os.path.exists('highway_model.pkl'):
    gdown.download(
        'https://drive.google.com/uc?id=1Hb0i83uGj5MWsmpKhiueYalVp-0v-hw6',
        'highway_model.pkl',
        quiet=False
    )

app = FastAPI()
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


GEMINI_AVAILABLE = False
gemini_model = None
try:
    import google.generativeai as genai
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        GEMINI_AVAILABLE = True
        logging.info(" Gemini fallback initialized (gemini-2.0-flash)")
except ImportError:
    logging.warning(" google-generativeai not installed. Gemini fallback disabled.")
    logging.warning("   Install with: pip install google-generativeai")
except Exception as e:
    logging.warning(f" Gemini init failed: {e}")

city_model = joblib.load('city_model.pkl')
highway_model = joblib.load('highway_model.pkl')


tavily_client = None
if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    logging.info("Tavily search initialized")
elif not TAVILY_AVAILABLE:
    logging.warning("Tavily not installed. Install with: pip install tavily-python")
else:
    logging.warning("TAVILY_API_KEY not set. Search will not work.")


def load_local_db():
    global local_cars_db
    if os.path.exists(LOCAL_DB_PATH):
        try:
            with open(LOCAL_DB_PATH, 'r', encoding='utf-8') as f:
                local_cars_db = json.load(f)
            logging.info(f" Loaded {len(local_cars_db)} cars from local DB")
        except Exception as e:
            logging.error(f" Failed to load local DB: {e}")
            local_cars_db = []
    else:
        logging.warning(f"{LOCAL_DB_PATH} not found. Starting with empty DB.")
        local_cars_db = []

def find_in_local_db(make: str, model: str, cc: Optional[float] = None) -> Optional[dict]:
    make_lower = make.lower().strip()
    model_lower = model.lower().strip()
    logging.info(f"Searching local DB for: {make} {model} (cc={cc})")
    
   
    cc_in_liters = None
    if cc is not None:
        if cc > 10:
            cc_in_liters = round(cc / 1000.0, 2)
        else:
            cc_in_liters = round(cc, 2)
    
    for car in local_cars_db:
        car_make = car.get('make', '').lower().strip()
        car_model = car.get('model', '').lower().strip()
     
        if car_make == make_lower and car_model == model_lower:
            if cc_in_liters is not None:  
                car_cc = car.get('engine_displacement_liters', 0)
                if abs(car_cc - cc_in_liters) < 0.05:  
                    logging.info(f"Found EXACT match in Local DB: {make} {model} {cc_in_liters}L")
                    return {
                        "engine_displacement_liters": car.get('engine_displacement_liters'),
                        "engine_cylinders": car.get('engine_cylinders', 4),
                        "drive": car.get('drive', 'FWD'),
                        "fuel_type": car.get('fuel_type', 'Regular Gasoline'),
                        "turbocharger": car.get('turbocharger', False)
                    }
            else:
                return {
                    "engine_displacement_liters": car.get('engine_displacement_liters'),
                    "engine_cylinders": car.get('engine_cylinders', 4),
                    "drive": car.get('drive', 'FWD'),
                    "fuel_type": car.get('fuel_type', 'Regular Gasoline'),
                    "turbocharger": car.get('turbocharger', False)
                }
    
    logging.info(f"Not found in Local DB: {make} {model}")
    return None

def save_to_local_db(make: str, model: str, year: int, specs: dict):
    new_car = {
        "make": make.strip(),
        "model": model.strip(),
        "engine_displacement_liters": specs.get('engine_displacement_liters'),
        "engine_cylinders": specs.get('engine_cylinders', 4),
        "drive": specs.get('drive', 'FWD'),
        "fuel_type": specs.get('fuel_type', 'Regular Gasoline'),
        "turbocharger": specs.get('turbocharger', False)
    }
    
    for car in local_cars_db:
        if (car['make'].lower().strip() == make.lower().strip() and 
            car['model'].lower().strip() == model.lower().strip() and
            car['engine_displacement_liters'] == new_car['engine_displacement_liters']):
            logging.info(f"Car already exists in Local DB: {make} {model}")
            return
        
    local_cars_db.append(new_car)
    try:
        with open(LOCAL_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(local_cars_db, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Saved {make} {model} to local DB file.")
    except Exception as e:
        logging.error(f"❌ Failed to write to local DB file: {e}")
    
load_local_db()


MAIN_SYSTEM_PROMPT = """
You are an expert automotive data extraction assistant specializing in the Egyptian and Middle Eastern automotive market. 

Your task is to extract the technical specifications of a vehicle from the provided search results based on the user's input (Make, Model, and Year).

CRITICAL RULES:
1. MARKET SPECIFICITY: You MUST prioritize specifications for cars sold in Egypt and the Middle East. DO NOT default to US or European specifications.
2. BEST-SELLING DEFAULT: If multiple variants exist, you MUST default to the most common or best-selling variant in the Egyptian market.
3. NO HALLUCINATION: If a specific detail cannot be found in the search results, you MUST return `null` for that key. DO NOT guess.
4. STRICT OUTPUT: You MUST return the output ONLY as a valid, raw JSON object. Do not include any text or markdown formatting outside the JSON object.

FIELD EXTRACTION GUIDE (look for BOTH English AND Arabic synonyms):
- engine_displacement_liters:
    English: "engine displacement", "engine size", "engine capacity", "liters", "L", "cc"
    Arabic: "سعة المحرك", "سعة الموتور", "سي سي", "سى سى", "قدرة المحرك", "لتر", "لترات"
- engine_cylinders:
    English: "cylinders", "V4", "V6", "V8", "inline-4", "I4"
    Arabic: "أسطوانات", "سلندر", "سلندرات", "عدد السلندرات"
- drive:
    English: "FWD", "RWD", "AWD", "4WD", "front-wheel", "rear-wheel", "4x4"
    Arabic: "دفع أمامي", "دفع خلفي", "دفع رباعي", "4×4", "دفع كلي"
- fuel_type:
    English: "gasoline", "petrol", "diesel", "hybrid"
    Arabic: "بنزين", "سولار", "ديزل", "هايبرد", "وقود"
    
    ⚠️ CRITICAL - FUEL TYPE MAPPING (Egyptian → US Model categories):
    The downstream ML model ONLY accepts these 4 exact string values.
    You MUST convert any Egyptian fuel type to one of them:
    
    - "بنزين 80" / "80 أوكتان" / "80 octane" / "regular" → "Regular Gasoline"
    - "بنزين 92" / "92 أوكتان" / "92 octane" / "midgrade" → "Midgrade Gasoline"
    - "بنزين 95" / "95 أوكتان" / "95 octane" / "premium" / "بنزين 95 أوكتان" → "Premium Gasoline"
    - "سولار" / "ديزل" / "diesel" → "Diesel"
    
    If the source mentions multiple options, pick the one the car RECOMMENDS
    (e.g., "يستخدم بنزين 92 أو 95" → use "Midgrade Gasoline" as default).
- turbocharger:
    English (has turbo): "turbo", "turbocharged", "T-GDI", "twin-turbo"
    English (no turbo): "naturally aspirated", "NA"
    Arabic (has turbo): "توربو", "شاحن هواء", "تيربو"
    Arabic (no turbo): "تنفس طبيعي", "غير توربو"
    
    ⚠️ DEFAULT RULE for Egyptian market:
    The MAJORITY of cars sold in Egypt are naturally aspirated (no turbo).
    If the search results DO NOT explicitly mention "turbo" or "توربو",
    you MUST return `false` (not null). Only return `true` if you find
    explicit evidence of a turbocharger.

UNIT CONVERSION RULES:
- If source says "1600cc" or "1600 cc" → return 1.6
- If source says "1.6L" → return 1.6
- If source says "سعة المحرك 1600 سي سي" → return 1.6
- If source says "1.6 لتر" → return 1.6

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

Look for keywords in BOTH English and Arabic:
- English: "engine displacement", "engine size", "engine capacity", "liters", "L", "cc"
- Arabic: "سعة المحرك", "سعة الموتور", "سي سي", "سى سى", "قدرة المحرك", "لتر", "لترات"

UNIT CONVERSION:
- "1600cc" or "1600 cc" → 1.6
- "1.6L" → 1.6
- "سعة المحرك 1600 سي سي" → 1.6
- "1.6 لتر" → 1.6

CRITICAL RULES: Prioritize Egyptian market specs. NO hallucinations. Use null if not found.
Respond ONLY with valid JSON: {"engine_displacement_liters": <float or null>}
"""

TURBO_PROMPT = """
You are an expert in the Egyptian automotive market.
Determine ONLY if this car has a Turbocharger from the search results.

Look for keywords in BOTH English and Arabic:
- English (has turbo): "turbo", "turbocharged", "T-GDI", "twin-turbo"
- English (no turbo): "naturally aspirated", "NA"
- Arabic (has turbo): "توربو", "شاحن هواء", "تيربو"
- Arabic (no turbo): "تنفس طبيعي", "غير توربو"

CRITICAL RULES: Prioritize Egyptian market specs. NO hallucinations. Use null if not found.
Respond ONLY with valid JSON: {"turbocharger": <boolean or null>}
"""

def _tavily_search(query: str) -> str:
    if not tavily_client:
        return ""
    try:
        logging.info(f" Tavily search: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        context_parts = []
        if response.get("answer"):
            context_parts.append(f"[AI Summary]: {response['answer']}")
        for r in response.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")
            context_parts.append(f"Title: {title}\nContent: {content}")
        return "\n\n".join(context_parts)
    except Exception as e:
        logging.error(f"Tavily search failed: {e}\n{traceback.format_exc()}")
        return ""


def search_web(query: str) -> str:
    return _tavily_search(query)


def multi_query_search(make: str, model: str, year: int) -> str:
    queries = [
        f'"{make} {model}" {year} engine displacement cc Egypt specifications',
        f'{make} {model} {year} مواصفات المحرك سعة سي سي مصر',
        f'{make} {model} {year} سعة الموتور سلندرات توربو مصر',
        f'{make} {model} {year} fuel consumption L/100km specs',
    ]

    all_contexts = []
    for q in queries:
        ctx = search_web(q)
        if ctx:
            all_contexts.append(ctx)
        if len(all_contexts) >= 2:  
            break
    return "\n\n=====\n\n".join(all_contexts) if all_contexts else ""


# Retry Mechanism ---

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def _call_groq_json(system_prompt: str, user_msg: str):
    """Calls Groq for JSON extraction with automatic retry on Rate Limit."""
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def _call_groq_recommendations(prompt: str):
    """Calls Groq for recommendations with automatic retry on Rate Limit."""
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def _call_gemini_recommendations(prompt: str):
    """Calls Gemini for recommendations with automatic retry on Rate Limit."""
    return gemini_model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 600,
        }
    )

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def _call_gemini_for_json(system_prompt: str, user_msg: str) -> str:
    """Calls Gemini for JSON extraction with automatic retry on Rate Limit."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available")
    
    full_prompt = f"{system_prompt}\n\n---\n\n{user_msg}\n\nReturn ONLY valid JSON, no markdown."
    response = gemini_model.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
    )
    return response.text


def extract_specs_with_llm(
    make: str,
    model: str,
    year: int,
    context: str,
    system_prompt: str,
    expect_field: Optional[str] = None,
) -> dict:

    try:
        if not context:
            logging.warning(f"No context provided to LLM (field={expect_field})")
            return {}

        user_msg = f"""
Target Car:
- Make: {make}
- Model: {model}
- Year: {year}

Search Results:
{context}

Extract the specifications for THIS specific car ONLY.
Return ONLY a valid JSON object as instructed.
""".strip()

        logging.info(f" Sending to LLM (expecting: {expect_field or 'all specs'})...")
        
        content = None
        
        try:
            logging.info(" Calling Groq (with retry)...")
            response = _call_groq_json(system_prompt, user_msg)  
            content = response.choices[0].message.content
            logging.info("✅ Groq responded")
        except Exception as groq_err:
            logging.warning(f"Groq failed after retries: {type(groq_err).__name__}")
            
            if GEMINI_AVAILABLE:
                logging.info("🔄 Falling back to Gemini (with retry)...")
                try:
                    content = _call_gemini_for_json(system_prompt, user_msg)  
                    logging.info("✅ Gemini responded")
                except Exception as gemini_err:
                    logging.error(f"Gemini also failed after retries: {gemini_err}")
                    return {}
            else:
                logging.error(f" No fallback available: {groq_err}")
                return {}
            
        if not content:
            logging.error("No content from either LLM")
            return {}
        
        logging.info(f"LLM response: {content[:500]}")
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse failed: {e}\nRaw: {content[:300] if 'content' in locals() else 'N/A'}")
        return {}
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}\n{traceback.format_exc()}")
        return {}


def get_car_specs(make: str, model: str, year: int, cc: Optional[float] = None) -> Optional[dict]:

    logging.info(f" Getting specs for {make} {model} {year}, cc={cc}")
    
    local_specs = find_in_local_db(make, model, cc)
    if local_specs:
        logging.info("Using Local DB (no API calls needed)")
        if cc is not None:
            if cc > 10:  
                local_specs["engine_displacement_liters"] = round(cc / 1000.0, 2)
            else:  
                local_specs["engine_displacement_liters"] = round(cc, 2)
        
        return local_specs
    
    
    logging.info("️ Not in Local DB. Searching via LLM...")
    
    context = multi_query_search(make, model, year)
    logging.info(f"Combined context length: {len(context)}")
    
    if not context:
        logging.error(">>> ROOT CAUSE: All searches returned empty")
        context = search_web(f"{make} {model} {year} specifications")
    if not context:
        logging.error(">>> Still empty. Cannot proceed without search context.")
        return None
    specs = extract_specs_with_llm(make, model, year, context, MAIN_SYSTEM_PROMPT, expect_field="all specs")
    if not specs:
        logging.error("Failed to extract specs from main search")
        return None
    logging.info(f"Initial specs from LLM: {specs}")
    
    if cc is not None:
        if cc > 10:  
            specs["engine_displacement_liters"] = round(cc / 1000.0, 2)
            logging.info(f"Using user-provided cc: {cc}cc = {specs['engine_displacement_liters']}L")
        else:  
            specs["engine_displacement_liters"] = round(cc, 2)
            logging.info(f"Using user-provided liters: {specs['engine_displacement_liters']}L")
    
    elif specs.get("engine_displacement_liters") is None:
        logging.warning("Engine displacement is null. Triggering DEDICATED search...")
        disp_query = f'"{make} {model}" {year} engine size displacement liters cc Egypt'
        disp_context = search_web(disp_query)
        disp_result = extract_specs_with_llm(
            make, model, year, disp_context, DISP_PROMPT,
            expect_field="engine_displacement_liters"
        )
        if disp_result and disp_result.get("engine_displacement_liters") is not None:
            specs["engine_displacement_liters"] = disp_result["engine_displacement_liters"]
            logging.info(f"Found engine displacement: {specs['engine_displacement_liters']}L")
   
    if specs.get("engine_displacement_liters") is not None:
        save_to_local_db(make, model, year, specs)
    return specs


def apply_safety_net(specs: dict) -> Optional[dict]:
    logging.info("🛡️ Applying Safety Net...")

    if specs.get("engine_displacement_liters") is None:
        logging.error(">>> engine_displacement_liters is None after all attempts")
        return None
    if specs.get("drive") is None:
        specs["drive"] = "FWD"
        logging.info("Set drive to default: FWD")
    if specs.get("turbocharger") is None:
        specs["turbocharger"] = False
        logging.info("Set turbocharger to default: False")
    if specs.get("engine_cylinders") is None:
        specs["engine_cylinders"] = 4
        logging.info("Set engine_cylinders to default: 4")
    if specs.get("fuel_type") is None:
        specs["fuel_type"] = "Gasoline"
        logging.info("Set fuel_type to default: Gasoline")

    return specs


class TripInput(BaseModel):
    make: str
    model: str
    year: int
    road_type: str
    temperature: float
    ac_on: bool
    cc: Optional[float] = None
    from_location: str  # required - used for distance calc + recommendations
    to_location: str    # required - used for distance calc + recommendations


def adjust_consumption(mpg, temperature, ac_on):
    liters = 235.21 / mpg
    if ac_on:
        if temperature > 35:
            liters *= 1.20
        else:
            liters *= 1.08
    liters *= 1.20
    return round(liters, 2)


def _normalize_fuel_type(fuel: str) -> str:
    if not fuel:
        return "Regular Gasoline"  # default
    
    fuel_lower = fuel.lower().strip()
    
    if any(kw in fuel_lower for kw in ["diesel", "سولار", "ديزل"]):
        return "Diesel"
    if any(kw in fuel_lower for kw in ["95", "premium", "سوبر", "بنزين 95"]):
        return "Premium Gasoline"
    if any(kw in fuel_lower for kw in ["92", "midgrade", "متوسط", "بنزين 92"]):
        return "Midgrade Gasoline"
    if any(kw in fuel_lower for kw in ["80", "regular", "بنزين 80", "gasoline", "petrol", "بنزين"]):
        return "Regular Gasoline"
    
    logging.warning(f"Unknown fuel_type '{fuel}', defaulting to Regular Gasoline")
    return "Regular Gasoline"


def predict_consumption(trip: TripInput, specs: dict) -> float:
    drive = specs.get("drive", "FWD")
    drive_4wd = 1 if drive == "4WD" else 0
    drive_rwd = 1 if drive == "RWD" else 0

    fuel = specs.get("fuel_type", "Gasoline")
    fuel = _normalize_fuel_type(fuel)
    logging.info(f"Normalized fuel_type: {fuel}")
    
    fuel_diesel = 1 if fuel == "Diesel" else 0
    fuel_midgrade = 1 if fuel == "Midgrade Gasoline" else 0
    fuel_premium = 1 if fuel == "Premium Gasoline" else 0

    current_year = 2026 
    car_age = current_year - trip.year
    logging.info(f"Car year: {trip.year} → car_age: {car_age}")

    input_data = pd.DataFrame({
        "Engine Displacement": [specs["engine_displacement_liters"]],
        "Engine Cylinders": [specs.get("engine_cylinders", 4)],
        "Turbocharger": [1 if specs.get("turbocharger") else 0],
        "car_age": [car_age],  # ← was "Year" before
        "Drive_4WD": [drive_4wd],
        "Drive_RWD": [drive_rwd],
        "Fuel Type 1_Diesel": [fuel_diesel],
        "Fuel Type 1_Midgrade Gasoline": [fuel_midgrade],
        "Fuel Type 1_Premium Gasoline": [fuel_premium],
    })

    if trip.road_type.lower().strip() == "city":
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
        
    route_context = _get_route_context(trip.from_location, trip.to_location, trip.road_type)

    prompt = f"""
You are an expert automotive advisor with deep knowledge of Egyptian roads, fuel efficiency, and road safety.

 TASK: Provide EXACTLY 4 actionable, personalized recommendations to help the driver reduce fuel consumption and drive safely on this specific trip.

 TRIP DETAILS:
- Car: {trip.make} {trip.model} (Year: {trip.year}, Age: {age_category})
- Engine: {specs.get('engine_displacement_liters')}L, {specs.get('engine_cylinders', 4)} cylinders
- Turbo: {"Yes" if specs.get('turbocharger') else "No"}
- Fuel Type: {specs.get('fuel_type')}
- Road Type: {trip.road_type}
- Temperature: {trip.temperature}°C
- AC: {"On" if trip.ac_on else "Off"}
- Predicted Consumption: {consumption} L/100km
- Route: {trip.from_location} → {trip.to_location}

🛣️ ROUTE CONTEXT (analysis of the road):
{route_context}

📝 REQUIREMENTS:
1. Provide EXACTLY 4 recommendations - not more, not less.
2. Each recommendation MUST be directly tied to the specific trip conditions above (car model, road type, weather, route, AC usage).
3. Focus on ACTIONABLE fuel-saving tips that consider:
   - The actual road conditions between {trip.from_location} and {trip.to_location}
   - The weather ({trip.temperature}°C) and AC usage
   - The car's age and engine specs
4. Be SPECIFIC to Egyptian driving conditions (traffic patterns, road quality, weather).
5. Each tip should be 2-3 sentences maximum, practical and directly applicable.

🚫 DO NOT:
- Give generic advice like "drive smoothly" without context
- Repeat the same tip in different words
- Add introductions or conclusions (just the 4 tips)
- Number them as "Tip 1, Tip 2" - just use clear bullet points

📤 OUTPUT FORMAT:
• [First specific recommendation tied to route/weather/car]
• [Second specific recommendation]
• [Third specific recommendation]
• [Fourth specific recommendation]
"""
    try:
        message = _call_groq_recommendations(prompt)  # <--- هنا التعديل
        return message.choices[0].message.content
    except Exception as groq_err:
        logging.warning(f"Groq failed for recommendations: {type(groq_err).__name__}")
        if GEMINI_AVAILABLE:
            logging.info("🔄 Falling back to Gemini for recommendations...")
            try:
                response = _call_gemini_recommendations(prompt)  # <--- هنا التعديل
                return response.text
            except Exception as gemini_err:
                logging.error(f" Gemini also failed for recommendations: {gemini_err}")
                return "Unable to generate recommendations at this time. Please try again later."
        else:
            logging.error(f" No fallback available: {groq_err}")
            return "Unable to generate recommendations at this time. Please try again later."


def _get_route_context(from_loc: str, to_loc: str, road_type: str) -> str:
    from_loc = from_loc.strip().lower()
    to_loc = to_loc.strip().lower()
    
    long_distance_routes = [
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
    ]
    
    urban_routes = [
        ("cairo", "giza"), ("giza", "cairo"),
        ("nasr city", "maadi"), ("maadi", "nasr city"),
        ("nasr city", "new cairo"), ("new cairo", "nasr city"),
        ("cairo", "6th october"), ("6th october", "cairo"),
        ("cairo", "new cairo"), ("new cairo", "cairo"),
    ]
    
    route_tuple = (from_loc, to_loc)
    
    if route_tuple in long_distance_routes:
        return f"""- This is a LONG-DISTANCE intercity trip in Egypt (~200+ km)
- Road is typically a desert highway with: open road, fewer stops, higher speeds (90-120 km/h)
- Possible challenges: crosswinds, sand on road, limited fuel stations, fatigue
- Weather consideration: {road_type} driving with potential temperature variations
- Best fuel-saving strategy: maintain steady highway speed, use cruise control if available"""
    
    elif route_tuple in urban_routes:
        return f"""- This is an URBAN trip within Egyptian city traffic
- Road has: heavy traffic, frequent stops, traffic lights, low speeds (10-40 km/h)
- Possible challenges: stop-and-go traffic, idling, frequent braking
- AC usage has higher impact in city driving due to idling
- Best fuel-saving strategy: avoid aggressive acceleration, anticipate stops, minimize idling"""
    
    elif from_loc == to_loc:
        return f"""- This appears to be a LOCAL trip within {from_loc.title()}
- Likely city driving with frequent stops and traffic
- Short distance, but fuel efficiency still matters for daily commutes"""
    
    else:
        if road_type.lower() == "city":
            return f"""- Trip from {from_loc.title()} to {to_loc.title()}
- City driving conditions: traffic, stops, lower speeds
- Focus on urban fuel-saving techniques"""
        else:
            return f"""- Trip from {from_loc.title()} to {to_loc.title()}
- Highway driving conditions: open road, higher speeds
- Focus on highway fuel-saving techniques (steady speed, aerodynamics)"""


@app.post("/predict")
def predict(trip: TripInput):
    logging.info(f"📡 Predict request: {trip.make} {trip.model} {trip.year}, cc={trip.cc}")


    specs = get_car_specs(trip.make, trip.model, trip.year, trip.cc)
    if not specs:
        return {
            "status": "error",
            "message": "Failed to fetch car data. Please try again."
        }

    safe_specs = apply_safety_net(specs)
    if safe_specs is None:
        return {
            "status": "missing_critical_data",
            "message": f"تم العثور على سيارة {trip.make} {trip.model}، لكن لم نتمكن من تأكيد سعة المحرك بدقة.",
            "missing_fields": ["engine_displacement_liters"],
            "suggested_options": [1.4, 1.6, 2.0]
        }

    consumption = predict_consumption(trip, safe_specs)
    logging.info(f"Consumption calculated: {consumption} L/100km")

    recommendations = get_recommendations(trip, consumption, safe_specs)
    logging.info("Recommendations generated")

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
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tavily_available": tavily_client is not None,
        "gemini_available": GEMINI_AVAILABLE,
    }


@app.get("/ping")
def ping():
    """Ultra-simple health check - no dependencies."""
    return {"pong": True}


@app.get("/debug/env")
def debug_env():
    """Check what env vars are loaded (without exposing values)."""
    return {
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
        "groq_key_prefix": (os.getenv("GROQ_API_KEY") or "")[:6] + "...",
        "tavily_key_prefix": (os.getenv("TAVILY_API_KEY") or "")[:6] + "...",
        "cwd": os.getcwd(),
        "env_file_exists": os.path.exists(".env"),
    }


@app.get("/debug/models")
def debug_models():
    """Check if the ML models are loaded correctly."""
    return {
        "city_model_loaded": city_model is not None,
        "highway_model_loaded": highway_model is not None,
        "city_model_type": str(type(city_model).__name__),
        "highway_model_type": str(type(highway_model).__name__),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
