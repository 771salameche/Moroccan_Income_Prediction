#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit web application for the Moroccan income prediction model.

This application provides:
1. An intuitive interface for entering prediction inputs
2. Connection to the prediction API
3. Clear display of prediction results
4. Data visualizations about Moroccan income distribution
5. Explanations of the model and features
6. Form validation
7. Feature importance for each prediction
8. Comparison of different scenarios
9. Responsive design for different devices
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from typing import Dict, List, Optional, Union, Any
import time
import os
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Moroccan Income Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_URL = "http://127.0.0.1:8000"  # Default local API URL
API_KEY = "development_key"  # Default development key

# Income statistics from data generation
REVENU_MOYEN = 21949
REVENU_MOYEN_URBAIN = 26988
REVENU_MOYEN_RURAL = 12862

# Feature options (based on API model)
GENDER_OPTIONS = ["Homme", "Femme"]
MILIEU_OPTIONS = ["Urbain", "Rural"]
EDUCATION_OPTIONS = ["Sans", "Fondamental", "Secondaire", "Supérieur"]
MARITAL_OPTIONS = ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve"]
CATEGORY_OPTIONS = [
    "Cadre supérieur",
    "Cadre moyen",
    "Employé",
    "Ouvrier qualifié",
    "Ouvrier non qualifié",
    "Agriculteur",
    "Commerçant",
    "Artisan",
    "Profession libérale",
    "Retraité",
    "Sans emploi"
]
SECTOR_OPTIONS = ["Public", "Privé", "Informel", "Sans emploi"]
AGE_CATEGORY_OPTIONS = ["Jeune", "Adulte", "Senior"]

# Custom CSS
def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .confidence-interval {
        font-size: 1rem;
        color: #616161;
        text-align: center;
        margin-top: 5px;
    }
    .feature-importance-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .tooltip-icon {
        color: #9e9e9e;
        font-size: 16px;
        margin-left: 5px;
    }
    .comparison-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .language-selector {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #9e9e9e;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def format_currency(amount):
    """Format amount as Moroccan Dirhams."""
    return f"{amount:,.2f} MAD"

def get_age_category(age):
    """Determine age category based on age."""
    if age < 30:
        return "Jeune"
    elif age < 60:
        return "Adulte"
    else:
        return "Senior"

def call_api(endpoint, data, api_key=API_KEY):
    """Call the prediction API."""
    headers = {"X-API-Key": api_key}
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def predict_income(features, return_confidence=True, confidence_level=0.95, return_explanation=True):
    """
    Make a prediction using the API or mock prediction if API is not available.

    This function will try to call the API first, and if that fails, it will use a mock prediction.
    """
    # Try to call the API first
    try:
        data = {
            "features": features,
            "return_confidence": return_confidence,
            "confidence_level": confidence_level,
            "return_explanation": return_explanation
        }
        result = call_api("predict", data)
        if result:
            return result
    except:
        pass

    # If API call fails, use mock prediction
    return mock_predict_income(features, return_confidence, confidence_level, return_explanation)

def mock_predict_income(features, return_confidence=True, confidence_level=0.95, return_explanation=True):
    """Generate a mock prediction when the API is not available."""
    import uuid
    from datetime import datetime
    import random

    # Base income calculation based on features
    base_income = 20000  # Base income in MAD

    # Adjust for education
    education_factor = {
        "Sans": 0.7,
        "Fondamental": 0.9,
        "Secondaire": 1.1,
        "Supérieur": 1.4
    }
    base_income *= education_factor.get(features["Niveau_Education"], 1.0)

    # Adjust for urban/rural
    if features["Milieu"] == "Urbain":
        base_income *= 1.3
    else:
        base_income *= 0.8

    # Adjust for experience
    experience_factor = min(features["Annees_Experience"] * 0.03, 0.6)
    base_income *= (1 + experience_factor)

    # Adjust for gender (reflecting real-world disparities)
    if features["Sexe"] == "Homme":
        base_income *= 1.08
    else:
        base_income *= 0.92

    # Adjust for age
    if features["Age"] < 25:
        base_income *= 0.8
    elif features["Age"] < 35:
        base_income *= 1.0
    elif features["Age"] < 55:
        base_income *= 1.15
    else:
        base_income *= 0.95

    # Adjust for wealth indicators
    if features["Possession_Voiture"]:
        base_income *= 1.1
    if features["Possession_Logement"]:
        base_income *= 1.05
    if features["Possession_Terrain"]:
        base_income *= 1.08

    # Adjust for wealth index
    base_income *= (1 + features["Indice_Richesse"] * 0.05)

    # Add some randomness
    predicted_income = base_income * random.uniform(0.9, 1.1)

    # Create response
    response = {
        "predicted_income": predicted_income,
        "request_id": str(uuid.uuid4()),
        "prediction_time": datetime.now().isoformat()
    }

    # Add confidence interval if requested
    if return_confidence:
        margin = predicted_income * (1 - confidence_level) * 0.5
        response["confidence_interval"] = {
            "lower_bound": predicted_income - margin,
            "upper_bound": predicted_income + margin,
            "confidence_level": confidence_level
        }

    # Add feature importance if requested
    if return_explanation:
        # Create mock feature importance
        features_importance = [
            {"feature": "Niveau_Education", "importance": 0.35},
            {"feature": "Milieu", "importance": 0.25},
            {"feature": "Annees_Experience", "importance": 0.15},
            {"feature": "Age", "importance": 0.10},
            {"feature": "Sexe", "importance": 0.08},
            {"feature": "Indice_Richesse", "importance": 0.07},
            {"feature": "Possession_Voiture", "importance": 0.05},
            {"feature": "Possession_Logement", "importance": 0.04},
            {"feature": "Possession_Terrain", "importance": 0.03},
            {"feature": "Nb_Personnes_Charge", "importance": -0.02},
            {"feature": "Acces_Internet", "importance": 0.02},
            {"feature": "Secteur_Activite", "importance": 0.06},
            {"feature": "Etat_Matrimonial", "importance": 0.03},
            {"feature": "Categorie_Socioprofessionnelle", "importance": 0.12},
        ]
        response["feature_importance"] = features_importance

    return response

def get_model_info():
    """Get model information from the API or return mock info if API is not available."""
    try:
        response = requests.get(f"{API_URL}/model/info", headers={"X-API-Key": API_KEY})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        # Return mock model info
        return {
            "model_type": "GradientBoostingRegressor",
            "features": [
                'Age', 'Categorie_Age', 'Sexe', 'Milieu', 'Niveau_Education', 'Annees_Experience',
                'Etat_Matrimonial', 'Categorie_Socioprofessionnelle', 'Possession_Voiture',
                'Possession_Logement', 'Possession_Terrain', 'Nb_Personnes_Charge',
                'Secteur_Activite', 'Acces_Internet', 'Indice_Richesse', 'Education_Superieure'
            ],
            "training_date": "2023-05-13",
            "version": "1.0.0",
            "description": "Gradient Boosting model for predicting annual income of Moroccan individuals"
        }

# UI Components
def display_header():
    """Display the application header."""
    st.markdown('<div class="main-header">Moroccan Income Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
        Predict annual income based on demographic and socioeconomic factors
        </div>
        """,
        unsafe_allow_html=True
    )

def create_prediction_form():
    """Create the prediction input form."""
    with st.form("prediction_form"):
        st.markdown('<div class="sub-header">Enter Personal Information</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=15,
                max_value=100,
                value=35,
                help="Age in years (15-100)"
            )

            sexe = st.selectbox(
                "Gender",
                options=GENDER_OPTIONS,
                help="Gender of the individual"
            )

            milieu = st.selectbox(
                "Living Environment",
                options=MILIEU_OPTIONS,
                help="Urban or rural living environment"
            )

            niveau_education = st.selectbox(
                "Education Level",
                options=EDUCATION_OPTIONS,
                help="Highest level of education attained"
            )

            education_superieure = 1 if niveau_education == "Supérieur" else 0

        with col2:
            annees_experience = st.number_input(
                "Years of Experience",
                min_value=0.0,
                max_value=80.0,
                value=10.0,
                step=0.5,
                help="Years of work experience"
            )

            etat_matrimonial = st.selectbox(
                "Marital Status",
                options=MARITAL_OPTIONS,
                help="Current marital status"
            )

            categorie_socioprofessionnelle = st.selectbox(
                "Socio-professional Category",
                options=CATEGORY_OPTIONS,
                help="Professional category or occupation"
            )

            secteur_activite = st.selectbox(
                "Activity Sector",
                options=SECTOR_OPTIONS,
                help="Sector of employment"
            )

        with col3:
            possession_voiture = st.checkbox(
                "Car Ownership",
                value=True,
                help="Whether the individual owns a car"
            )

            possession_logement = st.checkbox(
                "Housing Ownership",
                value=True,
                help="Whether the individual owns their housing"
            )

            possession_terrain = st.checkbox(
                "Land Ownership",
                value=False,
                help="Whether the individual owns land"
            )

            nb_personnes_charge = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=10,
                value=2,
                help="Number of people financially dependent on the individual"
            )

            acces_internet = st.checkbox(
                "Internet Access",
                value=True,
                help="Whether the individual has access to the internet"
            )

            indice_richesse = st.slider(
                "Wealth Index",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Index representing overall wealth (0-5)"
            )

        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                return_confidence = st.checkbox("Include confidence interval", value=True)
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    disabled=not return_confidence
                )
            with col2:
                return_explanation = st.checkbox("Show feature importance", value=True)

        # Submit button
        submitted = st.form_submit_button("Predict Income")

        # Prepare features dictionary if form is submitted
        if submitted:
            # Automatically determine age category
            categorie_age = get_age_category(age)

            features = {
                "Age": age,
                "Categorie_Age": categorie_age,
                "Sexe": sexe,
                "Milieu": milieu,
                "Niveau_Education": niveau_education,
                "Annees_Experience": annees_experience,
                "Etat_Matrimonial": etat_matrimonial,
                "Categorie_Socioprofessionnelle": categorie_socioprofessionnelle,
                "Possession_Voiture": possession_voiture,
                "Possession_Logement": possession_logement,
                "Possession_Terrain": possession_terrain,
                "Nb_Personnes_Charge": nb_personnes_charge,
                "Secteur_Activite": secteur_activite,
                "Acces_Internet": acces_internet,
                "Indice_Richesse": indice_richesse,
                "Education_Superieure": education_superieure
            }

            return features, return_confidence, confidence_level, return_explanation

        return None, None, None, None

def display_prediction_result(result):
    """Display the prediction result."""
    if not result:
        return

    st.markdown('<div class="sub-header">Prediction Result</div>', unsafe_allow_html=True)

    # Main prediction card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # Predicted income
    st.markdown(
        f'<div class="prediction-value">{format_currency(result["predicted_income"])}</div>',
        unsafe_allow_html=True
    )

    # Confidence interval
    if "confidence_interval" in result:
        ci = result["confidence_interval"]
        st.markdown(
            f'<div class="confidence-interval">{ci["confidence_level"]*100:.0f}% Confidence Interval: '
            f'{format_currency(ci["lower_bound"])} - {format_currency(ci["upper_bound"])}</div>',
            unsafe_allow_html=True
        )

    # Request info
    st.markdown(
        f'<div style="font-size: 0.8rem; color: #9e9e9e; margin-top: 15px;">'
        f'Request ID: {result["request_id"]}<br>'
        f'Prediction Time: {result["prediction_time"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    if "feature_importance" in result and result["feature_importance"]:
        st.markdown('<div class="feature-importance-title">Feature Importance</div>', unsafe_allow_html=True)

        # Sort features by importance
        features = sorted(
            result["feature_importance"],
            key=lambda x: abs(x["importance"]),
            reverse=True
        )

        # Create a DataFrame for the chart
        df_importance = pd.DataFrame(features)

        # Create a horizontal bar chart
        fig = px.bar(
            df_importance,
            y="feature",
            x="importance",
            orientation="h",
            title="Impact on Prediction",
            labels={"importance": "Impact", "feature": "Feature"},
            color="importance",
            color_continuous_scale=["red", "gray", "blue"],
            range_color=[-max(abs(df_importance["importance"])), max(abs(df_importance["importance"]))],
        )

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Explanation of feature importance
        st.markdown(
            """
            <div style="font-size: 0.9rem; margin-top: 10px;">
            <b>How to interpret:</b> Blue bars indicate features that increase the predicted income,
            while red bars indicate features that decrease it. The length of each bar represents the
            magnitude of the impact.
            </div>
            """,
            unsafe_allow_html=True
        )

def create_income_distribution_visualization():
    """Create visualizations of Moroccan income distribution."""
    st.markdown('<div class="sub-header">Moroccan Income Distribution</div>', unsafe_allow_html=True)

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Overall Distribution", "Urban vs. Rural", "By Education Level"])

    with tab1:
        # Overall income distribution
        fig = go.Figure()

        # Generate sample data based on statistics
        urban_income = np.random.normal(REVENU_MOYEN_URBAIN, 15000, 1000)
        rural_income = np.random.normal(REVENU_MOYEN_RURAL, 7000, 500)
        all_income = np.concatenate([urban_income, rural_income])

        fig.add_trace(go.Histogram(
            x=all_income,
            nbinsx=50,
            name="Income Distribution",
            marker_color="#1E88E5",
            opacity=0.7
        ))

        fig.add_vline(
            x=REVENU_MOYEN,
            line_dash="dash",
            line_color="red",
            annotation_text=f"National Average: {format_currency(REVENU_MOYEN)}",
            annotation_position="top right"
        )

        fig.update_layout(
            title="Overall Income Distribution in Morocco",
            xaxis_title="Annual Income (MAD)",
            yaxis_title="Frequency",
            bargap=0.1,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("National Average", f"{REVENU_MOYEN:,.0f} MAD")
        with col2:
            st.metric("Urban Average", f"{REVENU_MOYEN_URBAIN:,.0f} MAD", f"+{REVENU_MOYEN_URBAIN-REVENU_MOYEN:,.0f}")
        with col3:
            st.metric("Rural Average", f"{REVENU_MOYEN_RURAL:,.0f} MAD", f"{REVENU_MOYEN_RURAL-REVENU_MOYEN:,.0f}")

    with tab2:
        # Urban vs. Rural comparison
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=urban_income,
            nbinsx=40,
            name="Urban",
            marker_color="#1E88E5",
            opacity=0.7
        ))

        fig.add_trace(go.Histogram(
            x=rural_income,
            nbinsx=40,
            name="Rural",
            marker_color="#43A047",
            opacity=0.7
        ))

        fig.add_vline(
            x=REVENU_MOYEN_URBAIN,
            line_dash="dash",
            line_color="#1565C0",
            annotation_text=f"Urban Average",
            annotation_position="top right"
        )

        fig.add_vline(
            x=REVENU_MOYEN_RURAL,
            line_dash="dash",
            line_color="#2E7D32",
            annotation_text=f"Rural Average",
            annotation_position="top right"
        )

        fig.update_layout(
            title="Urban vs. Rural Income Distribution",
            xaxis_title="Annual Income (MAD)",
            yaxis_title="Frequency",
            bargap=0.1,
            height=500,
            barmode="overlay"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Urban-Rural gap explanation
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <b>Urban-Rural Income Gap:</b> There is a significant income disparity between urban and rural areas in Morocco.
            Urban residents earn on average <b>2.1 times</b> more than their rural counterparts. This gap is attributed to:
            <ul>
                <li>Greater economic opportunities in cities</li>
                <li>Higher concentration of high-skilled jobs in urban areas</li>
                <li>Better access to education and professional development</li>
                <li>More developed infrastructure and services</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab3:
        # Income by education level
        education_levels = ["Sans", "Fondamental", "Secondaire", "Supérieur"]
        education_coeffs = [0.7, 0.9, 1.1, 1.4]
        avg_incomes = [REVENU_MOYEN * coeff for coeff in education_coeffs]

        fig = px.bar(
            x=education_levels,
            y=avg_incomes,
            labels={"x": "Education Level", "y": "Average Annual Income (MAD)"},
            title="Average Income by Education Level",
            color=avg_incomes,
            color_continuous_scale="Blues",
            text=[f"{income:,.0f} MAD" for income in avg_incomes]
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            height=500,
            yaxis=dict(range=[0, max(avg_incomes) * 1.1])
        )

        st.plotly_chart(fig, use_container_width=True)

        # Education impact explanation
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <b>Impact of Education:</b> Education level is one of the strongest predictors of income in Morocco.
            Individuals with higher education earn significantly more than those with little or no formal education.
            <ul>
                <li>Higher education (university degree) can double income compared to no formal education</li>
                <li>Secondary education provides approximately a 57% increase over no education</li>
                <li>Each additional level of education shows diminishing but significant returns</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_model_explanation():
    """Create explanation of the model and features."""
    st.markdown('<div class="sub-header">Model Explanation</div>', unsafe_allow_html=True)

    # Create tabs for different explanation sections
    tab1, tab2 = st.tabs(["Model Overview", "Feature Descriptions"])

    with tab1:
        st.markdown(
            """
            ### Moroccan Income Prediction Model

            This application uses a machine learning model to predict the annual income of Moroccan individuals
            based on demographic and socioeconomic factors. The model was trained on a comprehensive dataset
            of Moroccan income data.

            #### Model Type
            The prediction is powered by a **Gradient Boosting Regression** model, which is well-suited for
            capturing complex relationships between features and predicting numerical values like income.

            #### How It Works
            1. **Data Collection**: The model was trained on a dataset containing demographic and socioeconomic
               information of Moroccan individuals, along with their annual income.

            2. **Feature Engineering**: Raw data was processed to extract meaningful patterns and relationships.

            3. **Model Training**: The gradient boosting algorithm learned to predict income based on the
               provided features by minimizing prediction errors.

            4. **Validation**: The model was validated using cross-validation techniques to ensure its
               reliability and generalizability.

            5. **Prediction**: When you enter your information, the model processes it through the same
               pipeline used during training and generates a prediction.

            #### Confidence Intervals
            The model provides confidence intervals to indicate the range within which the actual income
            is likely to fall. This helps account for uncertainty in the predictions.

            #### Feature Importance
            The model can explain which factors most influenced a specific prediction, providing
            transparency into the decision-making process.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Feature Descriptions

            The model uses the following features to make predictions:

            #### Demographic Features
            - **Age**: Age in years (15-100)
            - **Categorie_Age**: Age category (Jeune, Adulte, Senior)
            - **Sexe**: Gender (Homme, Femme)
            - **Milieu**: Living environment (Urbain, Rural)
            - **Etat_Matrimonial**: Marital status (Célibataire, Marié(e), Divorcé(e), Veuf/Veuve)

            #### Education and Professional Features
            - **Niveau_Education**: Education level (Sans, Fondamental, Secondaire, Supérieur)
            - **Education_Superieure**: Whether the individual has higher education (0 or 1)
            - **Annees_Experience**: Years of work experience
            - **Categorie_Socioprofessionnelle**: Socio-professional category
            - **Secteur_Activite**: Activity sector (Public, Privé, Informel, Sans emploi)

            #### Economic Indicators
            - **Possession_Voiture**: Car ownership (True/False)
            - **Possession_Logement**: Housing ownership (True/False)
            - **Possession_Terrain**: Land ownership (True/False)
            - **Indice_Richesse**: Wealth index (0-5)

            #### Other Factors
            - **Nb_Personnes_Charge**: Number of dependents
            - **Acces_Internet**: Internet access (True/False)
            """
        )

        # Feature importance explanation
        st.markdown(
            """
            ### How Features Affect Income

            Based on the model, here are some key insights about how different factors affect income in Morocco:

            - **Education** has one of the strongest positive impacts on income, with higher education levels
              associated with significantly higher earnings.

            - **Urban residence** typically leads to higher income compared to rural areas, reflecting
              the economic opportunities available in cities.

            - **Professional category** strongly influences income, with skilled positions and management
              roles commanding higher salaries.

            - **Experience** generally has a positive relationship with income, though this effect may
              diminish after many years.

            - **Gender** can impact income, with men typically earning more than women in similar positions,
              reflecting gender disparities in the labor market.

            - **Wealth indicators** like car and property ownership are both predictors and results of
              higher income.
            """
        )

def create_scenario_comparison():
    """Create a feature to compare different prediction scenarios."""
    st.markdown('<div class="sub-header">Scenario Comparison</div>', unsafe_allow_html=True)

    # Check if there are saved scenarios in session state
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []

    # Display saved scenarios
    if st.session_state.scenarios:
        st.markdown("### Saved Scenarios")

        # Create columns for the scenarios
        cols = st.columns(min(len(st.session_state.scenarios), 3))

        # Display each scenario in a column
        for i, scenario in enumerate(st.session_state.scenarios):
            col_index = i % 3
            with cols[col_index]:
                st.markdown(f'<div class="comparison-card">', unsafe_allow_html=True)
                st.markdown(f"#### Scenario {i+1}")
                st.markdown(f"**Predicted Income:** {format_currency(scenario['result']['predicted_income'])}")

                # Display key features
                st.markdown("**Key Features:**")
                st.markdown(f"- Age: {scenario['features']['Age']}")
                st.markdown(f"- Gender: {scenario['features']['Sexe']}")
                st.markdown(f"- Education: {scenario['features']['Niveau_Education']}")
                st.markdown(f"- Environment: {scenario['features']['Milieu']}")
                st.markdown(f"- Experience: {scenario['features']['Annees_Experience']} years")

                # Delete button
                if st.button(f"Delete Scenario {i+1}", key=f"delete_{i}"):
                    st.session_state.scenarios.pop(i)
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        # Comparison visualization if there are at least 2 scenarios
        if len(st.session_state.scenarios) >= 2:
            st.markdown("### Visual Comparison")

            # Create data for comparison chart
            scenario_names = [f"Scenario {i+1}" for i in range(len(st.session_state.scenarios))]
            incomes = [scenario["result"]["predicted_income"] for scenario in st.session_state.scenarios]

            # Create bar chart
            fig = px.bar(
                x=scenario_names,
                y=incomes,
                labels={"x": "Scenario", "y": "Predicted Income (MAD)"},
                title="Income Comparison Across Scenarios",
                color=incomes,
                color_continuous_scale="Blues",
                text=[f"{income:,.0f} MAD" for income in incomes]
            )

            fig.update_traces(textposition="outside")

            fig.update_layout(
                height=400,
                yaxis=dict(range=[0, max(incomes) * 1.1])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Clear all button
            if st.button("Clear All Scenarios"):
                st.session_state.scenarios = []
                st.rerun()
    else:
        st.info(
            """
            No scenarios saved yet. Make predictions and save them to compare different scenarios.
            This feature allows you to compare how different demographic and socioeconomic factors
            affect predicted income.
            """
        )

def create_language_selector():
    """Create a language selector."""
    # This is a placeholder for future implementation of language support
    languages = ["English", "Français", "العربية"]
    selected_language = st.sidebar.selectbox("Language / Langue / اللغة", languages, index=0)

    # In a real implementation, this would change the UI language
    # For now, we'll just display a message
    if selected_language != "English":
        st.sidebar.info(
            "Full language support will be implemented in a future version. "
            "Currently, the application is available in English only."
        )

def main():
    """Main application function."""
    # Load custom CSS
    load_css()

    # Create sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/2c/Flag_of_Morocco.svg", width=200)
    st.sidebar.title("Moroccan Income Prediction")

    # Language selector
    create_language_selector()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Income Prediction", "Data Visualization", "Model Explanation", "Scenario Comparison"]
    )

    # API configuration
    with st.sidebar.expander("API Configuration"):
        global API_URL, API_KEY
        API_URL = st.text_input("API URL", value=API_URL)
        API_KEY = st.text_input("API Key", value=API_KEY, type="password")

    # Mock mode indicator
    try:
        # Try to connect to the API
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("✅ Connected to prediction API")
            using_mock = False
        else:
            st.sidebar.warning("⚠️ Using mock prediction model (API not available)")
            using_mock = True
    except:
        st.sidebar.warning("⚠️ Using mock prediction model (API not available)")
        using_mock = True

    # Display header
    display_header()

    # Page content
    if page == "Income Prediction":
        # Create prediction form
        features, return_confidence, confidence_level, return_explanation = create_prediction_form()

        # Make prediction if form is submitted
        if features:
            with st.spinner("Making prediction..."):
                result = predict_income(
                    features,
                    return_confidence=return_confidence,
                    confidence_level=confidence_level,
                    return_explanation=return_explanation
                )

            if result:
                # Display prediction result
                display_prediction_result(result)

                # Save scenario button
                if st.button("Save Scenario for Comparison"):
                    if "scenarios" not in st.session_state:
                        st.session_state.scenarios = []

                    # Add scenario to session state
                    st.session_state.scenarios.append({
                        "features": features,
                        "result": result
                    })

                    st.success("Scenario saved! Go to the Scenario Comparison page to compare it with others.")

    elif page == "Data Visualization":
        create_income_distribution_visualization()

    elif page == "Model Explanation":
        create_model_explanation()

    elif page == "Scenario Comparison":
        create_scenario_comparison()

    # Footer
    st.markdown(
        """
        <div class="footer">
        Moroccan Income Prediction App | Developed with Streamlit | Data based on Moroccan income statistics
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()