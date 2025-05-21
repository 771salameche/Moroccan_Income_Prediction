<div align="center">
  <h1 style="font-size: 3rem; font-weight: 700; color: #1a202c;">
    MOROCCAN INCOME PREDICTION
  </h1>
  <p style="font-size: 1.2rem; color: #4a5568; font-style: italic; margin-top: 1rem;">
    Predicting Moroccan incomes, empowering informed futures.
  </p>
  <div style="display: flex; justify-content: center; gap: 0.5rem; margin: 1.5rem 0;">
    <span style="background-color: #2d3748; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">
      <span>🔄 last commit</span>
      <span style="background-color: #3182ce; padding: 0.25rem 0.5rem; border-radius: 0.25rem; margin-left: 0.25rem;">today</span>
    </span>
    <span style="background-color: #2d3748; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">
      <span>📓 jupyter notebook</span>
      <span style="background-color: #3182ce; padding: 0.25rem 0.5rem; border-radius: 0.25rem; margin-left: 0.25rem;">97.8%</span>
    </span>
    <span style="background-color: #2d3748; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">
      <span>🌐 languages</span>
      <span style="background-color: #3182ce; padding: 0.25rem 0.5rem; border-radius: 0.25rem; margin-left: 0.25rem;">3</span>
    </span>
  </div>
  <h3 style="font-size: 1.5rem; color: #4a5568; margin-bottom: 1rem;">
    Built with the tools and technologies:
  </h3>
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin-bottom: 2rem;">
    <span style="background-color: #1a202c; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🔍 Sphinx</span>
    <span style="background-color: #e53e3e; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🌊 Streamlit</span>
    <span style="background-color: #ed8936; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🧠 scikit-learn</span>
    <span style="background-color: #38b2ac; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">⚡ FastAPI</span>
    <span style="background-color: #2b6cb0; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🔢 NumPy</span>
    <span style="background-color: #4299e1; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">📊 Pytest</span>
    <span style="background-color: #3182ce; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🐳 Docker</span>
    <span style="background-color: #2b6cb0; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🐍 Python</span>
    <span style="background-color: #4299e1; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🧮 SciPy</span>
    <span style="background-color: #2c5282; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🐼 pandas</span>
    <span style="background-color: #d53f8c; color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem;">🔍 Pydantic</span>
  </div>
</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Running Tests](#-running-tests)
- [Data Sources & Methodology](#-data-sources--methodology)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Project Overview

This project provides a comprehensive solution for predicting annual income for Moroccan individuals based on demographic and socioeconomic factors. It consists of two main components:

1. **Prediction API**: A FastAPI-based RESTful API that provides income prediction services
2. **Web Application**: A Streamlit-based interactive web interface for making predictions and visualizing data

The project aims to provide accurate income predictions that can be used for:
- Financial planning and decision-making
- Economic research and policy development
- Market analysis and segmentation
- Social welfare program targeting

## 🎯 Problem Statement

Income prediction is a critical challenge in Morocco, where economic disparities between urban and rural areas, as well as across different demographic groups, can be significant. Accurate income prediction models can help:

- **Individuals**: Make informed financial decisions and plan for their future
- **Businesses**: Better understand their customer base and market potential
- **Government**: Develop targeted policies and programs to address economic inequality
- **Researchers**: Study economic patterns and trends in the Moroccan context

Traditional methods of income assessment often rely on self-reporting, which can be inaccurate or incomplete. This project leverages machine learning to provide more objective and reliable income predictions based on a wide range of factors.

## ✨ Key Features

### Prediction API
- **Single Predictions**: Predict income for individual cases
- **Batch Predictions**: Process multiple predictions in one request
- **Confidence Intervals**: Provide statistical confidence ranges for predictions
- **Feature Importance**: Explain which factors most influenced each prediction
- **API Documentation**: Interactive Swagger UI for easy API exploration
- **Authentication**: API key authentication for secure access

### Web Application
- **Intuitive Interface**: User-friendly form for entering prediction inputs
- **Data Visualizations**: Interactive charts showing Moroccan income distribution
- **Model Explanations**: Clear descriptions of how the model works and what features it uses
- **Feature Importance**: Visual representation of how each factor affects predictions
- **Scenario Comparison**: Save and compare different prediction scenarios
- **Responsive Design**: Works well on desktop and mobile devices
- **Mock Prediction**: Fallback functionality when API is unavailable

## 📂 Project Structure

The project is organized into the following main directories:

```
Moroccan_Income_Prediction/
├── api/                  # FastAPI prediction API
│   ├── api.py            # Main API implementation
│   ├── client_example.py # Example client for API usage
│   ├── Dockerfile        # Docker configuration for API
│   └── run_api.py        # Script to run the API locally
├── app/                  # Streamlit web application
│   └── app.py            # Main web application implementation
├── data/                 # Data files
│   ├── processed/        # Processed data ready for modeling
│   └── raw/              # Raw data files
├── models/               # Trained models
│   └── final_moroccan_income_model.pkl  # Serialized prediction model
├── notebooks/            # Jupyter notebooks for analysis and development
│   ├── 01_data_generation.ipynb        # Data generation and exploration
│   ├── 02_data_preprocessing.ipynb     # Data cleaning and preprocessing
│   └── 03_model_development.ipynb      # Model training and evaluation
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   └── models/           # Model training and prediction scripts
│       └── predict_model.py  # Prediction functionality
└── tests/                # Test suite
    ├── test_api.py       # API tests
    └── test_model.py     # Model tests
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Setting Up a Virtual Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/771salameche/Moroccan_Income_Prediction.git
cd Moroccan_Income_Prediction

# Create and activate a virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### Installing Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Setting Up the API

```bash
# Run the API locally
python api/run_api.py
```

The API will be available at http://127.0.0.1:8000, with documentation at http://127.0.0.1:8000/docs.

### Setting Up the Web Application

```bash
# Run the Streamlit app
streamlit run app/app.py
```

The web application will be available at http://localhost:8501.

## 💻 Usage Examples

### Using the Prediction API

#### Single Prediction

```python
import requests
import json

# API endpoint and key
API_URL = "http://127.0.0.1:8000"
API_KEY = "development_key"  # Replace with your actual API key

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

# Make the prediction request
response = requests.post(
    f"{API_URL}/predict",
    json=data,
    headers={"X-API-Key": API_KEY}
)

# Print the result
if response.status_code == 200:
    result = response.json()
    print(f"Predicted Income: {result['predicted_income']:,.2f} MAD")

    if "confidence_interval" in result:
        ci = result["confidence_interval"]
        print(f"Confidence Interval ({ci['confidence_level']*100:.0f}%): "
              f"{ci['lower_bound']:,.2f} - {ci['upper_bound']:,.2f} MAD")

    if "feature_importance" in result:
        print("\nFeature Importance:")
        for feature in result["feature_importance"]:
            print(f"  {feature['feature']}: {feature['importance']:.4f}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

#### Batch Prediction

```python
# Batch prediction example
batch_data = {
    "features": [
        {
            "Age": 35,
            "Categorie_Age": "Adulte",
            "Sexe": "Homme",
            "Milieu": "Urbain",
            # ... other features ...
        },
        {
            "Age": 45,
            "Categorie_Age": "Adulte",
            "Sexe": "Femme",
            "Milieu": "Rural",
            # ... other features ...
        }
    ],
    "return_confidence": True,
    "confidence_level": 0.95
}

response = requests.post(
    f"{API_URL}/predict/batch",
    json=batch_data,
    headers={"X-API-Key": API_KEY}
)

# Process batch results
if response.status_code == 200:
    results = response.json()
    print(f"Processed {results['count']} predictions")
    for i, pred in enumerate(results["predictions"]):
        print(f"\nPrediction {i+1}: {pred['predicted_income']:,.2f} MAD")
```

### Using the Web Application

The Streamlit web application provides an intuitive graphical interface for making predictions:

1. **Navigate** to http://localhost:8501 after starting the app
2. **Fill in** the form with the individual's demographic and socioeconomic information
3. **Click** the "Predict Income" button to get a prediction
4. **View** the prediction results, including confidence intervals and feature importance
5. **Save** scenarios to compare different inputs
6. **Explore** data visualizations about Moroccan income distribution

<div align="center">
  <img src="images\Screenshot 2025-05-21 213812.png" alt="App Screenshot" style="max-width: 800px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

## 🧪 Running Tests

The project includes a comprehensive test suite to ensure functionality and accuracy:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_api.py
pytest tests/test_model.py

# Run tests with coverage report
pytest --cov=src --cov=api tests/
```

## 📊 Data Sources & Methodology

### Data Sources

The model is trained on a synthetic dataset that closely mirrors real-world Moroccan income distributions and demographic patterns. The dataset was generated based on:

- Official statistics from the Haut-Commissariat au Plan (HCP) of Morocco
- World Bank economic indicators for Morocco
- Academic research on income determinants in North African economies

### Key Statistics

- **Average Income**: 21,949 MAD annually
- **Urban Average**: 26,988 MAD annually
- **Rural Average**: 12,862 MAD annually
- **Income Distribution**: Approximately 71.8% of the population earns less than the average

### Methodology

1. **Data Generation**: Synthetic data was created to reflect real-world distributions
2. **Feature Engineering**: Raw features were processed and transformed
3. **Model Selection**: Various regression models were evaluated
4. **Hyperparameter Tuning**: Grid search was used to optimize model parameters
5. **Validation**: Cross-validation ensured model generalizability
6. **Uncertainty Estimation**: Confidence intervals were calibrated

## 📦 Dependencies

The project relies on the following main dependencies:

- **FastAPI**: Web API framework
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Pydantic**: Data validation
- **pytest**: Testing framework

For a complete list of dependencies, see the `requirements.txt` file.

## 🤝 Contributing

Contributions to the project are welcome! Here's how you can contribute:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Submit** a pull request

Please make sure to update tests as appropriate and follow the code style guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p style="font-size: 0.9rem; color: #718096;">
    Developed with ❤️ for Morocco | © 2025 | <a href="https://github.com/771salameche">771salameche</a>
  </p>
</div>
