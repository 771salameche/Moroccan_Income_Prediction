#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI implementation for the Moroccan income prediction model.

This module provides a RESTful API for:
1. Making single predictions
2. Making batch predictions
3. Getting model information
4. Health check endpoint
"""

import os
import sys
import time
import pickle
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field, validator, conlist
from pathlib import Path

# Add project root to path to import from src
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

# Import prediction module
from src.models.predict_model import (
    load_model,
    predict,
    validate_input_data,
    EXPECTED_FEATURES,
    fix_common_issues,
    explain_prediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_DIR, 'logs', 'api.log'), mode='a')
    ]
)
logger = logging.getLogger("moroccan-income-api")

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(PROJECT_DIR, 'logs'), exist_ok=True)

# API Key authentication
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "development_key")  # Default key for development
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Create FastAPI app
app = FastAPI(
    title="Moroccan Income Prediction API",
    description="API for predicting annual income of Moroccan individuals based on demographic and socioeconomic features",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and preprocessor at startup
model = None
preprocessor = None

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor at startup."""
    global model, preprocessor
    try:
        logger.info("Loading model and preprocessor...")
        model, preprocessor = load_model()
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # We'll continue running but predictions will fail until model is loaded

# Authentication dependency
async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests to the API."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Log request details
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")

    # Process the request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    logger.info(f"Request {request_id} completed in {process_time:.4f}s with status {response.status_code}")

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    return response

# Pydantic models for request validation
class IncomeFeatures(BaseModel):
    """Input features for income prediction."""
    Age: int = Field(..., description="Age in years", ge=15, le=100)
    Categorie_Age: str = Field(..., description="Age category (e.g., 'Jeune', 'Adulte', 'Senior')")
    Sexe: str = Field(..., description="Gender ('Homme' or 'Femme')")
    Milieu: str = Field(..., description="Living environment ('Urbain' or 'Rural')")
    Niveau_Education: str = Field(..., description="Education level")
    Annees_Experience: float = Field(..., description="Years of work experience", ge=0, le=80)
    Etat_Matrimonial: str = Field(..., description="Marital status")
    Categorie_Socioprofessionnelle: str = Field(..., description="Socio-professional category")
    Possession_Voiture: bool = Field(..., description="Car ownership")
    Possession_Logement: bool = Field(..., description="Housing ownership")
    Possession_Terrain: bool = Field(..., description="Land ownership")
    Nb_Personnes_Charge: int = Field(..., description="Number of dependents", ge=0, le=20)
    Secteur_Activite: str = Field(..., description="Activity sector")
    Acces_Internet: bool = Field(..., description="Internet access")
    Indice_Richesse: float = Field(..., description="Wealth index", ge=0, le=10)
    Education_Superieure: int = Field(..., description="Higher education (0 or 1)", ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "Age": 35,
                "Categorie_Age": "Adulte",
                "Sexe": "Homme",
                "Milieu": "Urbain",
                "Niveau_Education": "Supérieur",
                "Annees_Experience": 10.5,
                "Etat_Matrimonial": "Marié(e)",
                "Categorie_Socioprofessionnelle": "Cadre supérieur",
                "Possession_Voiture": True,
                "Possession_Logement": True,
                "Possession_Terrain": False,
                "Nb_Personnes_Charge": 2,
                "Secteur_Activite": "Privé",
                "Acces_Internet": True,
                "Indice_Richesse": 2.0,
                "Education_Superieure": 1
            }
        }

class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: IncomeFeatures
    return_confidence: bool = Field(True, description="Whether to return confidence intervals")
    confidence_level: float = Field(0.95, description="Confidence level (0-1)", ge=0.5, le=0.99)
    return_explanation: bool = Field(False, description="Whether to return feature importance explanation")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    features: conlist(IncomeFeatures, min_items=1, max_items=100)
    return_confidence: bool = Field(True, description="Whether to return confidence intervals")
    confidence_level: float = Field(0.95, description="Confidence level (0-1)", ge=0.5, le=0.99)
    return_explanation: bool = Field(False, description="Whether to return feature importance explanation")

class ConfidenceInterval(BaseModel):
    """Confidence interval for a prediction."""
    lower_bound: float
    upper_bound: float
    confidence_level: float

class FeatureImportance(BaseModel):
    """Feature importance for a prediction."""
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    """Single prediction response."""
    predicted_income: float
    confidence_interval: Optional[ConfidenceInterval] = None
    feature_importance: Optional[List[FeatureImportance]] = None
    request_id: str
    prediction_time: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    count: int
    request_id: str
    prediction_time: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    features: List[str]
    training_date: str
    version: str
    description: str

# API endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Moroccan Income Prediction API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css",
    )

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None and preprocessor is not None,
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(api_key: APIKey = Depends(get_api_key)):
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "features": EXPECTED_FEATURES,
        "training_date": "2023-05-13",  # Replace with actual training date
        "version": "1.0.0",
        "description": "Gradient Boosting model for predicting annual income of Moroccan individuals"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_income(
    request: PredictionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Predict annual income for a single individual.

    Returns the predicted income in Moroccan Dirhams (MAD) along with confidence intervals.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Check if model is loaded
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        features_dict = request.features.dict()

        # Make prediction
        prediction_result = predict(
            data=features_dict,
            model=model,
            preprocessor=preprocessor,
            return_confidence=request.return_confidence,
            confidence_level=request.confidence_level
        )

        # Prepare response
        response = {
            "predicted_income": float(prediction_result["predictions"][0]),
            "request_id": request_id,
            "prediction_time": datetime.now().isoformat()
        }

        # Add confidence interval if requested
        if request.return_confidence and "confidence_intervals" in prediction_result:
            ci = prediction_result["confidence_intervals"]
            response["confidence_interval"] = {
                "lower_bound": float(ci["lower_bounds"][0]),
                "upper_bound": float(ci["upper_bounds"][0]),
                "confidence_level": ci["confidence_level"]
            }

        # Add feature importance if requested
        if request.return_explanation:
            explanation = explain_prediction(
                data=features_dict,
                model=model,
                preprocessor=preprocessor
            )

            response["feature_importance"] = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(
                    explanation["feature_importance"]["features"],
                    explanation["feature_importance"]["importance_values"]
                )
            ]

        # Log prediction
        process_time = time.time() - start_time
        logger.info(f"Prediction {request_id} completed in {process_time:.4f}s: {response['predicted_income']:.2f} MAD")

        return response

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_income_batch(
    request: BatchPredictionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Predict annual income for multiple individuals.

    Returns the predicted incomes in Moroccan Dirhams (MAD) along with confidence intervals.
    Limited to 100 records per request.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Check if model is loaded
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert list of Pydantic models to DataFrame
        features_list = [item.dict() for item in request.features]
        features_df = pd.DataFrame(features_list)

        # Make batch prediction
        prediction_result = predict(
            data=features_df,
            model=model,
            preprocessor=preprocessor,
            return_confidence=request.return_confidence,
            confidence_level=request.confidence_level
        )

        # Prepare individual predictions
        predictions = []
        for i in range(len(features_list)):
            pred_response = {
                "predicted_income": float(prediction_result["predictions"][i]),
                "request_id": f"{request_id}-{i}",
                "prediction_time": datetime.now().isoformat()
            }

            # Add confidence interval if requested
            if request.return_confidence and "confidence_intervals" in prediction_result:
                ci = prediction_result["confidence_intervals"]
                pred_response["confidence_interval"] = {
                    "lower_bound": float(ci["lower_bounds"][i]),
                    "upper_bound": float(ci["upper_bounds"][i]),
                    "confidence_level": ci["confidence_level"]
                }

            predictions.append(pred_response)

        # Prepare batch response
        response = {
            "predictions": predictions,
            "count": len(predictions),
            "request_id": request_id,
            "prediction_time": datetime.now().isoformat()
        }

        # Log batch prediction
        process_time = time.time() - start_time
        logger.info(f"Batch prediction {request_id} with {len(predictions)} records completed in {process_time:.4f}s")

        return response

    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)