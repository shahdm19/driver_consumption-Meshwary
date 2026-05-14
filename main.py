from fastapi import FastAPI
import joblib
import pandas as pd
import gdown
import os

if not os.path.exists('city_model.pkl'):
   gdown.download('https://drive.google.com/uc?id=1zkL48TzAL2WfkaO7FymEJ49ttgrGm8Tw', 'city_model.pkl', quiet=False)

if not os.path.exists('highway_model.pkl'):
   gdown.download('https://drive.google.com/uc?id=1Hb0i83uGj5MWsmpKhiueYalVp-0v-hw6', 'highway_model.pkl', quiet=False)

app = FastAPI()

from pydantic import BaseModel

class TripInput(BaseModel):
    make: str
    model: str
    road_type: str
    temperature: float
    ac_on: bool


city_model = joblib.load('city_model.pkl')
highway_model = joblib.load('highway_model.pkl')
lookup_table = pd.read_csv('lookup_table.csv')
lookup_table = lookup_table.rename(columns={'Fuel Type': 'Fuel Type 1'})
lookup_table['Drive'] = lookup_table['Drive'].replace({
    '4-Wheel Drive': '4WD',
    'All-Wheel Drive': '4WD',
    '4-Wheel or All-Wheel Drive': '4WD',
    'Part-time 4-Wheel Drive': '4WD',
    'Front Wheel Drive': 'FWD',
    'Front-Wheel Drive': 'FWD',
    'Rear-Wheel Drive': 'RWD',
    '2-Wheel Drive': 'FWD'
})

def adjust_consumption(mpg, temperature, ac_on):
    liters = 235.21 / mpg
    if ac_on:
        if temperature > 35:
            liters *= 1.20
        else:
            liters *= 1.08
    liters *= 1.20
    return round(liters, 2)

def predict_consumption(make, model, road_type, temperature, ac_on):
    car = lookup_table[(lookup_table['Make'] == make) &
                       (lookup_table['Model'] == model)]

    if car.empty:
        return 'Car not found'

    car_age = 2026 - car['Year'].values[0]

    drive = car['Drive'].values[0]
    drive_4wd = 1 if drive == '4WD' else 0
    drive_rwd = 1 if drive == 'RWD' else 0

    fuel = car['Fuel Type 1'].values[0]
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    fuel_midgrade = 1 if fuel == 'Midgrade Gasoline' else 0
    fuel_premium = 1 if fuel == 'Premium Gasoline' else 0

    input_data = pd.DataFrame({
        'Engine Displacement': [car['Engine Displacement'].values[0]],
        'Engine Cylinders': [car['Engine Cylinders'].values[0]],
        'Turbocharger': [car['Turbocharger'].values[0]],
        'car_age': [car_age],
        'Drive_4WD': [drive_4wd],
        'Drive_RWD': [drive_rwd],
        'Fuel Type 1_Diesel': [fuel_diesel],
        'Fuel Type 1_Midgrade Gasoline': [fuel_midgrade],
        'Fuel Type 1_Premium Gasoline': [fuel_premium],
    })

    if road_type == 'city':
        mpg = city_model.predict(input_data)[0]
    else:
        mpg = highway_model.predict(input_data)[0]

    return adjust_consumption(mpg, temperature, ac_on)

@app.post("/predict")
def predict(trip: TripInput):
    result = predict_consumption(trip.make, trip.model, 
                                 trip.road_type, trip.temperature, trip.ac_on)
    return {"consumption_rate": result}