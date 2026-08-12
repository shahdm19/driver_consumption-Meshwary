# Meshwary — Fuel Consumption & Trip Cost Estimator

Meshwary predicts how much fuel a trip will cost, tailored to the Egyptian market. Give it a car (make, model, year), a route, and driving conditions, and it returns an estimated consumption rate along with a set of trip-specific driving recommendations.

## How It Works

1. **Car specs lookup.** The service first checks a local cache for the car's engine specs (displacement, cylinders, drive type, fuel type, turbocharger). If the car isn't cached, it searches the web (via Tavily) for the relevant specs and uses an LLM to extract structured data from the results — prioritizing Egyptian/Middle Eastern market variants over US or European defaults.
2. **Consumption prediction.** The extracted specs, along with trip conditions (road type, temperature, AC usage, car age), are fed into a pair of XGBoost regression models — one for city driving, one for highway driving — trained to R² = 0.965.
3. **Recommendations.** Based on the predicted consumption and route context (e.g. Cairo–Alexandria vs. inner-city traffic), an LLM generates four short, trip-specific tips for saving fuel and driving safely.
4. **Resilience.** LLM calls (spec extraction and recommendations) use Groq as the primary provider with automatic fallback to Gemini if Groq is unavailable or rate-limited.

A separate script keeps fuel prices current: it searches for the latest official Egyptian fuel prices (80/92/95 octane, diesel, natural gas), extracts them with an LLM, and falls back to the last known values for any price it can't confirm — so the app never serves stale or missing data.

## Tech Stack

- **API:** FastAPI
- **ML:** XGBoost (city/highway consumption models), scikit-learn pipeline, joblib
- **LLMs:** Groq (primary), Gemini (fallback)
- **Search:** Tavily (spec lookup), DuckDuckGo scraping (fuel prices)
- **Data:** Local JSON cache for known car specs and fuel prices

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Returns predicted fuel consumption and recommendations for a trip |
| `/health` | GET | Reports service status and integration availability |
| `/ping` | GET | Basic liveness check |

**Example request to `/predict`:**

```json
{
  "make": "Toyota",
  "model": "Corolla",
  "year": 2022,
  "road_type": "city",
  "temperature": 32,
  "ac_on": true,
  "from_location": "Cairo",
  "to_location": "Giza"
}
```

## Fuel Price Updates

`update_fuel_prices.py` runs independently of the main API and keeps `fuel_prices.json` up to date. It:

- Searches for the latest official fuel prices in Egypt
- Extracts them into a fixed schema using an LLM
- Runs a dedicated follow-up search for natural gas prices if missing
- Falls back to the previous known values for anything still unresolved
- Defaults the date field to today if no date can be extracted from search results

Intended to run on a schedule (e.g. a daily cron job) so the app's pricing data stays current without manual updates.

## Environment Variables

```
GROQ_API_KEY=
GEMINI_API_KEY=
TAVILY_API_KEY=
FUEL_PRICES_API_KEY=
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Deployment

Deployed via Docker on Hugging Face Spaces.

## Notes

This is a graduation project focused on solving a real, everyday problem for Egyptian drivers: knowing what a trip will actually cost in fuel before taking it, using locally relevant car data instead of generic international estimates.
