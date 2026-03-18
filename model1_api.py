# ============================================================
#  QUICKPRINT — MODEL 1 FASTAPI SERVICE
#  Serves the Peak Hour & Demand Forecasting model
#
#  HOW TO RUN:
#  1. pip install fastapi uvicorn joblib pandas numpy scikit-learn xgboost
#  2. Make sure quickprint_model1.pkl is in the same folder as this file
#  3. Run: uvicorn model1_api:app --reload
#  4. Open browser: http://localhost:8000/docs  (auto-generated UI to test)
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# ── Create the FastAPI app ───────────────────────────────────
app = FastAPI(
    title="QuickPrint — Model 1 API",
    description="Predicts demand (number of orders) for the next 1-hour slot at a print vendor.",
    version="1.0.0"
)
# FastAPI() = creates the web server app
# title, description, version = shows up in the auto docs at /docs


# ── Load the trained model when server starts ────────────────
MODEL_PATH = "quickprint_model1.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Make sure quickprint_model1.pkl is in the same folder as this file."
    )

model = joblib.load(MODEL_PATH)
# joblib.load() = loads the saved model from the .pkl file
# this runs ONCE when the server starts — not on every request
print(f"✓ Model loaded from {MODEL_PATH}")


# ── Define what the INPUT should look like ───────────────────
class DemandRequest(BaseModel):
    # BaseModel = pydantic class that validates incoming data
    # Field() = lets us add description and example values

    vendor_id: int = Field(
        ...,                          # ... means this field is required
        ge=1, le=5,                   # ge=greater or equal, le=less or equal
        description="Vendor ID (1=main building, 2=library, 3=hostel, 4=science, 5=canteen)",
        example=1
    )
    month: int = Field(
        ..., ge=1, le=12,
        description="Month number (1=January to 12=December)",
        example=4
    )
    day_of_week: int = Field(
        ..., ge=0, le=6,
        description="Day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)",
        example=0
    )
    hour_of_day: int = Field(
        ..., ge=8, le=20,
        description="Hour of day in 24hr format (8am to 8pm)",
        example=10
    )
    is_exam_period: int = Field(
        ..., ge=0, le=1,
        description="Is it exam period? (1=yes, 0=no)",
        example=1
    )
    is_weekday: int = Field(
        ..., ge=0, le=1,
        description="Is it a weekday? (1=yes, 0=no)",
        example=1
    )
    active_printers: int = Field(
        ..., ge=1, le=4,
        description="Number of printers currently active",
        example=3
    )
    orders_prev_1h: int = Field(
        ..., ge=0,
        description="Number of orders received in the previous 1 hour",
        example=15
    )
    orders_prev_3h: int = Field(
        ..., ge=0,
        description="Number of orders received in the previous 3 hours",
        example=38
    )


# ── Define what the OUTPUT will look like ────────────────────
class DemandResponse(BaseModel):
    predicted_orders: int = Field(description="Predicted number of orders in next 1 hour")
    demand_class: str     = Field(description="Demand level: Low / Medium / High")
    recommendation: str   = Field(description="What the vendor should do right now")
    peak_level: float     = Field(description="Peak level score from 0.0 to 1.0 (used by Model 2)")


# ── Helper function: demand class + recommendation ───────────
def get_demand_info(predicted_orders: int):
    # converts a number into a class label and advice

    if predicted_orders <= 5:
        demand_class   = "Low"
        recommendation = "Normal operations. 1-2 printers sufficient."
    elif predicted_orders <= 12:
        demand_class   = "Medium"
        recommendation = "Moderate demand. Keep 2-3 printers active."
    else:
        demand_class   = "High"
        recommendation = "Peak demand incoming! Activate all printers and alert staff."

    # peak_level: normalise to 0.0–1.0 scale
    # this number gets passed to Model 2 as 'predicted_demand'
    # we clip at 40 as that's our realistic max
    peak_level = round(min(predicted_orders / 40.0, 1.0), 3)

    return demand_class, recommendation, peak_level


# ── ROUTE 1: Health check ────────────────────────────────────
@app.get("/")
def root():
    # @app.get("/") = when someone visits http://localhost:8000/
    # this function runs and returns a response
    return {
        "service": "QuickPrint Model 1 API",
        "status" : "running",
        "message": "Go to /docs to test the API"
    }


# ── ROUTE 2: Health check endpoint ──────────────────────────
@app.get("/health")
def health():
    # simple ping to check if the server is alive
    # your web app can call this to verify the ML service is up
    return {"status": "ok", "model": "quickprint_model1.pkl"}


# ── ROUTE 3: Main prediction endpoint ───────────────────────
@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    # @app.post("/predict") = when someone sends a POST request to /predict
    # request: DemandRequest = FastAPI automatically validates the input
    # response_model = FastAPI validates and formats the output

    try:
        # Step 1: convert request into a pandas dataframe (1 row)
        input_data = pd.DataFrame([{
            "vendor_id"       : request.vendor_id,
            "month"           : request.month,
            "day_of_week"     : request.day_of_week,
            "hour_of_day"     : request.hour_of_day,
            "is_exam_period"  : request.is_exam_period,
            "is_weekday"      : request.is_weekday,
            "active_printers" : request.active_printers,
            "orders_prev_1h"  : request.orders_prev_1h,
            "orders_prev_3h"  : request.orders_prev_3h,
        }])

        # Step 2: run the model
        raw_prediction = model.predict(input_data)[0]
        # model.predict() returns a numpy array — [0] gets the first value

        # Step 3: round to a whole number (can't have 7.3 orders)
        predicted_orders = max(0, int(round(raw_prediction)))

        # Step 4: get demand class and recommendation
        demand_class, recommendation, peak_level = get_demand_info(predicted_orders)

        # Step 5: return the response
        return DemandResponse(
            predicted_orders = predicted_orders,
            demand_class     = demand_class,
            recommendation   = recommendation,
            peak_level       = peak_level
        )

    except Exception as e:
        # if anything goes wrong, return a proper error message
        # HTTPException = FastAPI's way of returning error responses
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── ROUTE 4: Batch prediction (multiple slots at once) ───────
@app.post("/predict/batch")
def predict_batch(requests: list[DemandRequest]):
    # lets you send multiple requests in one API call
    # useful for predicting the whole day at once

    if len(requests) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 requests per batch"
        )

    results = []
    for req in requests:
        input_data = pd.DataFrame([{
            "vendor_id"       : req.vendor_id,
            "month"           : req.month,
            "day_of_week"     : req.day_of_week,
            "hour_of_day"     : req.hour_of_day,
            "is_exam_period"  : req.is_exam_period,
            "is_weekday"      : req.is_weekday,
            "active_printers" : req.active_printers,
            "orders_prev_1h"  : req.orders_prev_1h,
            "orders_prev_3h"  : req.orders_prev_3h,
        }])

        raw             = model.predict(input_data)[0]
        predicted       = max(0, int(round(raw)))
        demand_class, recommendation, peak_level = get_demand_info(predicted)

        results.append({
            "hour_of_day"     : req.hour_of_day,
            "predicted_orders": predicted,
            "demand_class"    : demand_class,
            "recommendation"  : recommendation,
            "peak_level"      : peak_level
        })

    return {"total_slots": len(results), "predictions": results}


# ── ROUTE 5: Predict full day for a vendor ───────────────────
@app.get("/predict/day/{vendor_id}")
def predict_full_day(
    vendor_id: int,
    month: int        = 4,
    day_of_week: int  = 0,
    is_exam_period: int = 0
):
    # predict all hours 8am to 8pm for one vendor in one call
    # example: GET /predict/day/1?month=4&day_of_week=0&is_exam_period=1

    if vendor_id not in [1, 2, 3, 4, 5]:
        raise HTTPException(status_code=400, detail="vendor_id must be 1-5")

    results = []
    is_weekday = 1 if day_of_week < 5 else 0

    for hour in range(8, 21):
        # for day prediction, use simple rolling estimates
        prev_1h = 5 if hour == 8 else results[-1]['predicted_orders']
        prev_3h = prev_1h * 2

        input_data = pd.DataFrame([{
            "vendor_id"       : vendor_id,
            "month"           : month,
            "day_of_week"     : day_of_week,
            "hour_of_day"     : hour,
            "is_exam_period"  : is_exam_period,
            "is_weekday"      : is_weekday,
            "active_printers" : 2,
            "orders_prev_1h"  : prev_1h,
            "orders_prev_3h"  : prev_3h,
        }])

        raw       = model.predict(input_data)[0]
        predicted = max(0, int(round(raw)))
        demand_class, recommendation, peak_level = get_demand_info(predicted)

        results.append({
            "hour"            : f"{hour:02d}:00",
            "predicted_orders": predicted,
            "demand_class"    : demand_class,
            "peak_level"      : peak_level
        })

    # find the peak hour of the day
    peak_hour = max(results, key=lambda x: x['predicted_orders'])

    return {
        "vendor_id"   : vendor_id,
        "month"       : month,
        "day_of_week" : day_of_week,
        "is_exam_period": is_exam_period,
        "peak_hour"   : peak_hour['hour'],
        "hourly_forecast": results
    }
