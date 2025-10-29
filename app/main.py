from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from app.schemas import DeliveryRequest
import os

app = FastAPI(title="Food Delivery Time Prediction")

# Mount static files directory for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# Load model and preprocessor once
pipeline = PredictionPipeline()

def featch_features(data: DeliveryRequest) -> pd.DataFrame:
    """
    Return a DataFrame with all features from DeliveryRequest,
    including computed ones but excluding raw datetime fields.
    """
    features = {
        # --- Original input fields ---
        "Delivery_person_Age": data.Delivery_person_Age,
        "Delivery_person_Ratings": data.Delivery_person_Ratings,
        "Restaurant_latitude": data.Restaurant_latitude,
        "Restaurant_longitude": data.Restaurant_longitude,
        "Delivery_location_latitude": data.Delivery_location_latitude,
        "Delivery_location_longitude": data.Delivery_location_longitude,
        "Weatherconditions": data.Weatherconditions,
        "Road_traffic_density": data.Road_traffic_density,
        "Vehicle_condition": data.Vehicle_condition,
        "Type_of_order": data.Type_of_order,
        "Type_of_vehicle": data.Type_of_vehicle,
        "multiple_deliveries": data.multiple_deliveries,
        "Festival": data.Festival,
        "City": data.City,

        # --- Computed fields ---
        "prep_time(m)": data.prep_time_m,
        "order_hour": data.order_hour,
        "order_day_of_week": data.order_day_of_week,
        "is_weekend": data.is_weekend,
        "order_day": data.order_day,
        "order_week": data.order_week,
        "order_month": data.order_month,
        "distance_km": data.distance_km,
        "manhattan_km": data.manhattan_km,
        "distance_per_speed": data.distance_per_speed,
        "distance_ratio": data.distance_ratio,
        "rating_age_ratio": data.rating_age_ratio,
        "rating_vehicle": data.rating_vehicle,
    }

    return pd.DataFrame([features])

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the HTML form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_delivery_time(data: DeliveryRequest):
    """Predict Food Delivery Time from input data"""
    try:
        # log full payload including computed fields (helpful for debugging)
        logging.info(f"Received prediction request: {data.dict()}")

        # build DataFrame from the Pydantic model (uses computed properties)
        df = featch_features(data)

        # predict
        pred = pipeline.predict(df)[0]
        return {"prediction (minutes)": float(pred)}

    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")