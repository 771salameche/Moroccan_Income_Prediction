#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Moroccan income prediction API.

This module provides tests for:
1. API health check
2. Model information endpoint
3. Single prediction endpoint
4. Batch prediction endpoint
"""

import os
import sys
import json
import pytest
import requests
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

# Import API for testing
from api.api import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

# Test data
test_features = {
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

# Mock API key for testing
API_KEY = "development_key"
HEADERS = {"X-API-Key": API_KEY}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "api_version" in data
    assert "timestamp" in data

def test_model_info():
    """Test the model information endpoint."""
    response = client.get("/model/info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data
    assert "training_date" in data
    assert "version" in data
    assert "description" in data

def test_predict_single():
    """Test the single prediction endpoint."""
    request_data = {
        "features": test_features,
        "return_confidence": True,
        "confidence_level": 0.95,
        "return_explanation": False
    }
    
    response = client.post("/predict", json=request_data, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_income" in data
    assert isinstance(data["predicted_income"], float)
    assert "confidence_interval" in data
    assert "request_id" in data
    assert "prediction_time" in data

def test_predict_single_with_explanation():
    """Test the single prediction endpoint with feature importance."""
    request_data = {
        "features": test_features,
        "return_confidence": True,
        "confidence_level": 0.95,
        "return_explanation": True
    }
    
    response = client.post("/predict", json=request_data, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_income" in data
    assert "confidence_interval" in data
    assert "feature_importance" in data
    assert isinstance(data["feature_importance"], list)
    assert len(data["feature_importance"]) > 0
    assert "feature" in data["feature_importance"][0]
    assert "importance" in data["feature_importance"][0]

def test_predict_batch():
    """Test the batch prediction endpoint."""
    request_data = {
        "features": [test_features, test_features],  # Two identical records for testing
        "return_confidence": True,
        "confidence_level": 0.95,
        "return_explanation": False
    }
    
    response = client.post("/predict/batch", json=request_data, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 2
    assert "request_id" in data
    assert "prediction_time" in data
    
    # Check individual predictions
    for prediction in data["predictions"]:
        assert "predicted_income" in prediction
        assert isinstance(prediction["predicted_income"], float)
        assert "confidence_interval" in prediction
        assert "request_id" in prediction
        assert "prediction_time" in prediction

def test_authentication():
    """Test API authentication."""
    # Test with invalid API key
    invalid_headers = {"X-API-Key": "invalid_key"}
    response = client.get("/model/info", headers=invalid_headers)
    assert response.status_code == 401
    
    # Test without API key
    response = client.get("/model/info")
    assert response.status_code == 401

def test_invalid_input():
    """Test API with invalid input data."""
    # Missing required field
    invalid_features = test_features.copy()
    del invalid_features["Age"]
    
    request_data = {
        "features": invalid_features,
        "return_confidence": True,
        "confidence_level": 0.95
    }
    
    response = client.post("/predict", json=request_data, headers=HEADERS)
    assert response.status_code == 422  # Unprocessable Entity
    
    # Invalid value type
    invalid_features = test_features.copy()
    invalid_features["Age"] = "thirty-five"  # String instead of int
    
    request_data = {
        "features": invalid_features,
        "return_confidence": True,
        "confidence_level": 0.95
    }
    
    response = client.post("/predict", json=request_data, headers=HEADERS)
    assert response.status_code == 422  # Unprocessable Entity
    
    # Value out of range
    invalid_features = test_features.copy()
    invalid_features["Age"] = 150  # Out of range
    
    request_data = {
        "features": invalid_features,
        "return_confidence": True,
        "confidence_level": 0.95
    }
    
    response = client.post("/predict", json=request_data, headers=HEADERS)
    assert response.status_code == 422  # Unprocessable Entity

if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
