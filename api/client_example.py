#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example client for the Moroccan income prediction API.

This script demonstrates how to:
1. Make a health check request
2. Get model information
3. Make a single prediction
4. Make a batch prediction
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from pprint import pprint

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example client for the Moroccan income prediction API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", type=str, default="development_key", help="API key")
    return parser.parse_args()

def health_check(base_url):
    """Check API health."""
    print("\n=== Health Check ===")
    response = requests.get(f"{base_url}/health")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        pprint(response.json())
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def get_model_info(base_url, api_key):
    """Get model information."""
    print("\n=== Model Information ===")
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{base_url}/model/info", headers=headers)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        pprint(response.json())
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def make_single_prediction(base_url, api_key):
    """Make a single prediction."""
    print("\n=== Single Prediction ===")
    headers = {"X-API-Key": api_key}
    
    # Example input data
    data = {
        "features": {
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
        },
        "return_confidence": True,
        "confidence_level": 0.95,
        "return_explanation": True
    }
    
    response = requests.post(f"{base_url}/predict", json=data, headers=headers)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredicted Income: {result['predicted_income']:,.2f} MAD")
        
        if "confidence_interval" in result:
            ci = result["confidence_interval"]
            print(f"Confidence Interval ({ci['confidence_level']*100:.0f}%): "
                  f"{ci['lower_bound']:,.2f} - {ci['upper_bound']:,.2f} MAD")
        
        if "feature_importance" in result and result["feature_importance"]:
            print("\nFeature Importance:")
            for item in result["feature_importance"]:
                print(f"  {item['feature']}: {item['importance']:.4f}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def make_batch_prediction(base_url, api_key):
    """Make a batch prediction."""
    print("\n=== Batch Prediction ===")
    headers = {"X-API-Key": api_key}
    
    # Example input data for batch prediction
    data = {
        "features": [
            {
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
            },
            {
                "Age": 25,
                "Categorie_Age": "Jeune",
                "Sexe": "Femme",
                "Milieu": "Rural",
                "Niveau_Education": "Secondaire",
                "Annees_Experience": 3.0,
                "Etat_Matrimonial": "Célibataire",
                "Categorie_Socioprofessionnelle": "Employé",
                "Possession_Voiture": False,
                "Possession_Logement": False,
                "Possession_Terrain": False,
                "Nb_Personnes_Charge": 0,
                "Secteur_Activite": "Public",
                "Acces_Internet": True,
                "Indice_Richesse": 1.0,
                "Education_Superieure": 0
            }
        ],
        "return_confidence": True,
        "confidence_level": 0.95,
        "return_explanation": False
    }
    
    response = requests.post(f"{base_url}/predict/batch", json=data, headers=headers)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nNumber of predictions: {result['count']}")
        
        for i, prediction in enumerate(result["predictions"]):
            print(f"\nPrediction {i+1}:")
            print(f"  Predicted Income: {prediction['predicted_income']:,.2f} MAD")
            
            if "confidence_interval" in prediction:
                ci = prediction["confidence_interval"]
                print(f"  Confidence Interval ({ci['confidence_level']*100:.0f}%): "
                      f"{ci['lower_bound']:,.2f} - {ci['upper_bound']:,.2f} MAD")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def main():
    """Run the example client."""
    args = parse_args()
    base_url = args.url
    api_key = args.api_key
    
    print(f"Using API at {base_url}")
    
    # Check API health
    if not health_check(base_url):
        print("API health check failed. Exiting.")
        return
    
    # Get model information
    get_model_info(base_url, api_key)
    
    # Make a single prediction
    make_single_prediction(base_url, api_key)
    
    # Make a batch prediction
    make_batch_prediction(base_url, api_key)

if __name__ == "__main__":
    main()
