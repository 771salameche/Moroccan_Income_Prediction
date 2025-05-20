# Moroccan Income Prediction API

A RESTful API for predicting the annual income of Moroccan individuals based on demographic and socioeconomic features.

## Features

- **RESTful API**: Implements a clean, RESTful API design
- **Model Loading**: Loads the trained model at startup
- **Input Validation**: Validates input data with Pydantic schemas
- **Prediction Endpoints**: Processes both single and batch prediction requests
- **Confidence Intervals**: Returns prediction confidence intervals
- **Feature Importance**: Provides SHAP-based feature importance explanations
- **Error Handling**: Implements proper error handling with appropriate status codes
- **Documentation**: Includes Swagger/OpenAPI documentation
- **Request Logging**: Logs all requests and responses
- **Authentication**: Implements API key authentication
- **Docker Support**: Includes Dockerfile for containerization

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the API Locally

1. Start the API server:
   ```
   python api/run_api.py
   ```

   This will start the API on http://127.0.0.1:8000 by default.

2. Access the API documentation:
   - Open http://127.0.0.1:8000/docs in your browser

### API Endpoints

- **GET /health**: Health check endpoint
- **GET /model/info**: Get model information (requires authentication)
- **POST /predict**: Make a single prediction (requires authentication)
- **POST /predict/batch**: Make batch predictions (requires authentication)

### Authentication

The API uses API key authentication. Include the API key in the `X-API-Key` header:

```
X-API-Key: your-api-key
```

For development, the default API key is `development_key`.

### Example Client

An example client is provided in `api/client_example.py`:

```
python api/client_example.py --url http://localhost:8000 --api-key development_key
```

## Docker Deployment

1. Build the Docker image:
   ```
   docker build -t moroccan-income-api -f api/Dockerfile .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 -e API_KEY=your-api-key moroccan-income-api
   ```

## Testing

Run the tests with pytest:

```
pytest tests/test_api.py
```

## Input Features

The API expects the following features for prediction:

| Feature                       | Type    | Description                                |
|-------------------------------|---------|--------------------------------------------|
| Age                           | int     | Age in years                               |
| Categorie_Age                 | string  | Age category (e.g., 'Jeune', 'Adulte')     |
| Sexe                          | string  | Gender ('Homme' or 'Femme')                |
| Milieu                        | string  | Living environment ('Urbain' or 'Rural')   |
| Niveau_Education              | string  | Education level                            |
| Annees_Experience             | float   | Years of work experience                   |
| Etat_Matrimonial              | string  | Marital status                             |
| Categorie_Socioprofessionnelle| string  | Socio-professional category                |
| Possession_Voiture            | boolean | Car ownership                              |
| Possession_Logement           | boolean | Housing ownership                          |
| Possession_Terrain            | boolean | Land ownership                             |
| Nb_Personnes_Charge           | int     | Number of dependents                       |
| Secteur_Activite              | string  | Activity sector                            |
| Acces_Internet                | boolean | Internet access                            |
| Indice_Richesse               | float   | Wealth index                               |
| Education_Superieure          | int     | Higher education (0 or 1)                  |

## Example Request

```json
{
  "features": {
    "Age": 35,
    "Categorie_Age": "Adulte",
    "Sexe": "Homme",
    "Milieu": "Urbain",
    "Niveau_Education": "Supérieur",
    "Annees_Experience": 10.5,
    "Etat_Matrimonial": "Marié(e)",
    "Categorie_Socioprofessionnelle": "Cadre supérieur",
    "Possession_Voiture": true,
    "Possession_Logement": true,
    "Possession_Terrain": false,
    "Nb_Personnes_Charge": 2,
    "Secteur_Activite": "Privé",
    "Acces_Internet": true,
    "Indice_Richesse": 2.0,
    "Education_Superieure": 1
  },
  "return_confidence": true,
  "confidence_level": 0.95,
  "return_explanation": true
}
```

## Example Response

```json
{
  "predicted_income": 120000.0,
  "confidence_interval": {
    "lower_bound": 100000.0,
    "upper_bound": 140000.0,
    "confidence_level": 0.95
  },
  "feature_importance": [
    {
      "feature": "Niveau_Education",
      "importance": 0.35
    },
    {
      "feature": "Annees_Experience",
      "importance": 0.25
    },
    ...
  ],
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction_time": "2023-05-20T12:34:56.789Z"
}
```

## License

[MIT License](LICENSE)
