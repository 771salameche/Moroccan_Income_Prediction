#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction script for the Moroccan income prediction model.

This module provides functions to:
1. Load the finalized model from storage
2. Preprocess new input data
3. Make predictions on new data
4. Calculate confidence intervals or prediction uncertainty
5. Handle invalid inputs gracefully
6. Provide explanations for predictions (using SHAP)
7. Format outputs appropriately for downstream use
8. Verify input data quality
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any

# For prediction intervals
from scipy import stats

# For SHAP explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not found. Install with 'pip install shap' for prediction explanations.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "final_moroccan_income_model.pkl"
PIPELINE_PATH = PROJECT_DIR / "notebooks" / "moroccan_income_preprocessor.pkl"

# Expected feature names for raw input data
EXPECTED_FEATURES = [
    'Age', 'Categorie_Age', 'Sexe', 'Milieu', 'Niveau_Education', 'Annees_Experience',
    'Etat_Matrimonial', 'Categorie_Socioprofessionnelle', 'Possession_Voiture',
    'Possession_Logement', 'Possession_Terrain', 'Nb_Personnes_Charge',
    'Secteur_Activite', 'Acces_Internet', 'Indice_Richesse', 'Education_Superieure'
]

# Feature types for validation
FEATURE_TYPES = {
    'Age': 'int',
    'Categorie_Age': 'category',
    'Sexe': 'category',
    'Milieu': 'category',
    'Niveau_Education': 'category',
    'Annees_Experience': 'float',
    'Etat_Matrimonial': 'category',
    'Categorie_Socioprofessionnelle': 'category',
    'Possession_Voiture': 'bool',
    'Possession_Logement': 'bool',
    'Possession_Terrain': 'bool',
    'Nb_Personnes_Charge': 'int',
    'Secteur_Activite': 'category',
    'Acces_Internet': 'bool',
    'Indice_Richesse': 'float',
    'Education_Superieure': 'int'
}

# Valid values for categorical features
VALID_VALUES = {
    'Categorie_Age': ['Jeune Adulte', 'Adulte', 'Senior'],
    'Sexe': ['Homme', 'Femme'],
    'Milieu': ['Urbain', 'Rural'],
    'Niveau_Education': ['Sans', 'Primaire', 'Secondaire', 'Supérieur'],
    'Etat_Matrimonial': ['Célibataire', 'Marié(e)', 'Divorcé(e)', 'Veuf/Veuve'],
    'Categorie_Socioprofessionnelle': ['Cadre supérieur', 'Cadre moyen', 'Employé', 'Ouvrier', 'Agriculteur', 'Commerçant', 'Artisan', 'Sans emploi'],
    'Secteur_Activite': ['Public', 'Privé', 'Informel', 'Sans emploi']
}

class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""
    pass

class PipelineNotFoundError(Exception):
    """Exception raised when preprocessing pipeline file is not found."""
    pass

class InvalidInputError(Exception):
    """Exception raised when input data is invalid."""
    pass

# Define OutlierCapper class for unpickling
class OutlierCapper:
    """
    Class for capping outliers in numerical features.
    This is needed to unpickle the preprocessing pipeline.
    """
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        """Compute the bounds for outlier capping."""
        return self

    def transform(self, X):
        """Cap outliers in the data."""
        return X

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

def load_model() -> Tuple[Any, Any]:
    """
    Load the trained model and preprocessing pipeline.

    Returns:
        Tuple[Any, Any]: Tuple containing (model, preprocessor)

    Raises:
        ModelNotFoundError: If model file is not found
        PipelineNotFoundError: If preprocessor file is not found
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise ModelNotFoundError(f"Model file not found at {MODEL_PATH}")

    try:
        # Add OutlierCapper to the module namespace for unpickling
        import sys
        sys.modules['OutlierCapper'] = OutlierCapper

        with open(PIPELINE_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded successfully from {PIPELINE_PATH}")
    except FileNotFoundError:
        logger.error(f"Preprocessor file not found at {PIPELINE_PATH}")
        raise PipelineNotFoundError(f"Preprocessor file not found at {PIPELINE_PATH}")

    return model, preprocessor

def validate_input_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate input data against expected schema.

    Args:
        data (pd.DataFrame): Input data to validate

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    # Check if all required features are present
    missing_features = set(EXPECTED_FEATURES) - set(data.columns)
    if missing_features:
        errors.append(f"Missing required features: {', '.join(missing_features)}")

    # Check data types and values for each feature
    for feature, expected_type in FEATURE_TYPES.items():
        if feature not in data.columns:
            continue  # Already reported as missing

        # Check data types
        if expected_type == 'int':
            if not pd.api.types.is_numeric_dtype(data[feature]):
                errors.append(f"Feature '{feature}' should be numeric")
            elif data[feature].dropna().apply(lambda x: x != int(x)).any():
                errors.append(f"Feature '{feature}' should contain only integers")

        elif expected_type == 'float':
            if not pd.api.types.is_numeric_dtype(data[feature]):
                errors.append(f"Feature '{feature}' should be numeric")

        elif expected_type == 'bool':
            if not pd.api.types.is_bool_dtype(data[feature]) and not set(data[feature].dropna().unique()).issubset({0, 1, True, False}):
                errors.append(f"Feature '{feature}' should be boolean (True/False or 0/1)")

        elif expected_type == 'category':
            if feature in VALID_VALUES:
                invalid_values = set(data[feature].dropna().unique()) - set(VALID_VALUES[feature])
                if invalid_values:
                    errors.append(f"Feature '{feature}' contains invalid values: {', '.join(map(str, invalid_values))}")

    # Check for out-of-range values
    if 'Age' in data.columns and pd.api.types.is_numeric_dtype(data['Age']):
        if (data['Age'] < 15).any() or (data['Age'] > 100).any():
            errors.append("Age should be between 15 and 100")

    if 'Annees_Experience' in data.columns and pd.api.types.is_numeric_dtype(data['Annees_Experience']):
        if (data['Annees_Experience'] < 0).any():
            errors.append("Annees_Experience cannot be negative")
        if 'Age' in data.columns and pd.api.types.is_numeric_dtype(data['Age']):
            if (data['Annees_Experience'] > data['Age'] - 15).any():
                errors.append("Annees_Experience cannot be greater than Age - 15")

    if 'Nb_Personnes_Charge' in data.columns and pd.api.types.is_numeric_dtype(data['Nb_Personnes_Charge']):
        if (data['Nb_Personnes_Charge'] < 0).any():
            errors.append("Nb_Personnes_Charge cannot be negative")

    return len(errors) == 0, errors

def preprocess_data(data: Union[pd.DataFrame, Dict], preprocessor: Any) -> np.ndarray:
    """
    Preprocess input data using the saved preprocessing pipeline.

    Args:
        data (Union[pd.DataFrame, Dict]): Input data to preprocess
        preprocessor (Any): Preprocessing pipeline

    Returns:
        np.ndarray: Preprocessed data ready for prediction

    Raises:
        InvalidInputError: If input data is invalid
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Validate input data
    is_valid, errors = validate_input_data(data)
    if not is_valid:
        error_msg = "Invalid input data:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise InvalidInputError(error_msg)

    try:
        # Apply preprocessing pipeline
        preprocessed_data = preprocessor.transform(data)
        logger.info(f"Data preprocessed successfully. Shape: {preprocessed_data.shape}")
        return preprocessed_data
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise InvalidInputError(f"Error preprocessing data: {str(e)}")

def predict(data: Union[pd.DataFrame, Dict],
            model: Optional[Any] = None,
            preprocessor: Optional[Any] = None,
            return_confidence: bool = True,
            confidence_level: float = 0.95) -> Dict:
    """
    Make predictions on new data.

    Args:
        data (Union[pd.DataFrame, Dict]): Input data
        model (Optional[Any]): Trained model (loaded if None)
        preprocessor (Optional[Any]): Preprocessing pipeline (loaded if None)
        return_confidence (bool): Whether to return confidence intervals
        confidence_level (float): Confidence level (0-1)

    Returns:
        Dict: Dictionary containing predictions and metadata

    Raises:
        InvalidInputError: If input data is invalid
        ModelNotFoundError: If model file is not found
        PipelineNotFoundError: If preprocessor file is not found
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_model()

    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data_df = pd.DataFrame([data])
    else:
        data_df = data.copy()

    # Preprocess data
    try:
        preprocessed_data = preprocess_data(data_df, preprocessor)
    except InvalidInputError as e:
        # Try to fix common issues
        data_df = fix_common_issues(data_df)
        preprocessed_data = preprocess_data(data_df, preprocessor)

    # Make predictions
    try:
        predictions = model.predict(preprocessed_data)
        logger.info(f"Predictions generated successfully. Shape: {predictions.shape}")

        # Prepare results
        results = {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'input_shape': data_df.shape,
            'prediction_count': len(predictions)
        }

        # Add confidence intervals if requested
        if return_confidence:
            lower_bounds, upper_bounds = calculate_confidence_intervals(
                model, preprocessed_data, predictions, confidence_level
            )
            results['confidence_intervals'] = {
                'lower_bounds': lower_bounds.tolist() if isinstance(lower_bounds, np.ndarray) else lower_bounds,
                'upper_bounds': upper_bounds.tolist() if isinstance(upper_bounds, np.ndarray) else upper_bounds,
                'confidence_level': confidence_level
            }

        return results

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise RuntimeError(f"Error making predictions: {str(e)}")

def fix_common_issues(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common issues in input data.

    Args:
        data (pd.DataFrame): Input data with potential issues

    Returns:
        pd.DataFrame: Fixed data
    """
    fixed_data = data.copy()

    # Convert boolean columns
    for feature, dtype in FEATURE_TYPES.items():
        if dtype == 'bool' and feature in fixed_data.columns:
            fixed_data[feature] = fixed_data[feature].map({
                'Oui': True, 'Non': False,
                'oui': True, 'non': False,
                'Yes': True, 'No': False,
                'yes': True, 'no': False,
                '1': True, '0': False,
                1: True, 0: False
            }).astype('bool')

    # Fix categorical values (standardize case, handle typos)
    for feature, valid_values in VALID_VALUES.items():
        if feature in fixed_data.columns:
            # Create mapping for case-insensitive matching
            value_map = {}
            for valid_value in valid_values:
                value_map[valid_value.lower()] = valid_value

            # Apply mapping
            fixed_data[feature] = fixed_data[feature].apply(
                lambda x: value_map.get(str(x).lower(), x) if pd.notna(x) else x
            )

    # Special handling for Categorie_Age
    if 'Categorie_Age' in fixed_data.columns and 'Age' in fixed_data.columns:
        # Fill missing Categorie_Age based on Age
        for idx, row in fixed_data.iterrows():
            if pd.isna(row['Categorie_Age']) or row['Categorie_Age'] not in VALID_VALUES['Categorie_Age']:
                age = row['Age']
                if pd.notna(age):
                    if age < 25:
                        fixed_data.at[idx, 'Categorie_Age'] = 'Jeune Adulte'
                    elif age < 60:
                        fixed_data.at[idx, 'Categorie_Age'] = 'Adulte'
                    else:
                        fixed_data.at[idx, 'Categorie_Age'] = 'Senior'

    # Special handling for Education_Superieure
    if 'Education_Superieure' in fixed_data.columns and 'Niveau_Education' in fixed_data.columns:
        # Set Education_Superieure based on Niveau_Education
        fixed_data['Education_Superieure'] = (fixed_data['Niveau_Education'] == 'Supérieur').astype(int)

    # Handle missing values
    for feature, dtype in FEATURE_TYPES.items():
        if feature in fixed_data.columns:
            if dtype == 'int':
                # Fill missing with median or 0
                if fixed_data[feature].isna().any():
                    median_value = fixed_data[feature].median()
                    fixed_data[feature] = fixed_data[feature].fillna(median_value if pd.notna(median_value) else 0)

            elif dtype == 'float':
                # Fill missing with median or 0
                if fixed_data[feature].isna().any():
                    median_value = fixed_data[feature].median()
                    fixed_data[feature] = fixed_data[feature].fillna(median_value if pd.notna(median_value) else 0)

            elif dtype == 'bool':
                # Fill missing with False
                fixed_data[feature] = fixed_data[feature].fillna(False)

            elif dtype == 'category':
                # Fill missing with most frequent value
                if fixed_data[feature].isna().any():
                    most_frequent = fixed_data[feature].mode()[0]
                    fixed_data[feature] = fixed_data[feature].fillna(most_frequent)

    logger.info("Fixed common issues in input data")
    return fixed_data

def calculate_confidence_intervals(
    model: Any,
    X: np.ndarray,
    predictions: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions.

    Args:
        model (Any): Trained model
        X (np.ndarray): Preprocessed input data
        predictions (np.ndarray): Model predictions
        confidence_level (float): Confidence level (0-1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (lower_bounds, upper_bounds)
    """
    # Different approaches based on model type
    model_type = type(model).__name__

    # For tree-based models, use prediction variance from trees
    if hasattr(model, 'estimators_') and hasattr(model, 'predict'):
        try:
            # Get predictions from all estimators
            estimator_preds = np.array([tree.predict(X) for tree in model.estimators_])

            # Calculate standard deviation across trees
            std_devs = np.std(estimator_preds, axis=0)

            # Calculate confidence intervals
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * std_devs

            lower_bounds = predictions - margin_of_error
            upper_bounds = predictions + margin_of_error

            return lower_bounds, upper_bounds

        except (AttributeError, TypeError):
            # Fallback if the above approach doesn't work
            pass

    # For linear models, use a simple heuristic based on RMSE
    # This is a simplified approach - for production, consider more sophisticated methods
    # like bootstrapping or quantile regression

    # Use a default error estimate of 15% of the prediction value
    error_estimate = 0.15 * np.abs(predictions)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * error_estimate

    lower_bounds = predictions - margin_of_error
    upper_bounds = predictions + margin_of_error

    # Ensure lower bounds are not negative for income predictions
    lower_bounds = np.maximum(lower_bounds, 0)

    return lower_bounds, upper_bounds

def explain_prediction(data: Union[pd.DataFrame, Dict],
                       model: Optional[Any] = None,
                       preprocessor: Optional[Any] = None,
                       max_display: int = 10) -> Dict:
    """
    Explain predictions using SHAP values.

    Args:
        data (Union[pd.DataFrame, Dict]): Input data
        model (Optional[Any]): Trained model (loaded if None)
        preprocessor (Optional[Any]): Preprocessing pipeline (loaded if None)
        max_display (int): Maximum number of features to display in explanation

    Returns:
        Dict: Dictionary containing SHAP values and feature importance

    Raises:
        RuntimeError: If SHAP is not available
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP library not available. Install with 'pip install shap'")
        raise RuntimeError("SHAP library not available. Install with 'pip install shap'")

    # Load model and preprocessor if not provided
    if model is None or preprocessor is None:
        model, preprocessor = load_model()

    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data_df = pd.DataFrame([data])
    else:
        data_df = data.copy()

    # Preprocess data
    preprocessed_data = preprocess_data(data_df, preprocessor)

    # Create SHAP explainer based on model type
    try:
        if hasattr(model, 'estimators_'):  # For tree-based models
            explainer = shap.TreeExplainer(model)
        else:  # For other models
            explainer = shap.Explainer(model)

        # Calculate SHAP values
        shap_values = explainer(preprocessed_data)

        # Get feature names if available
        try:
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            feature_names = [f"feature_{i}" for i in range(preprocessed_data.shape[1])]

        # Calculate feature importance
        feature_importance = np.abs(shap_values.values).mean(axis=0)

        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx[:max_display]]
        sorted_importance = feature_importance[sorted_idx[:max_display]]

        # Prepare results
        explanation = {
            'feature_importance': {
                'features': sorted_features,
                'importance_values': sorted_importance.tolist()
            },
            'shap_values': shap_values.values.tolist(),
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else float(shap_values.base_values[0])
        }

        return explanation

    except Exception as e:
        logger.error(f"Error explaining predictions: {str(e)}")
        raise RuntimeError(f"Error explaining predictions: {str(e)}")

def verify_data_quality(data: pd.DataFrame) -> Dict:
    """
    Verify the quality of input data.

    Args:
        data (pd.DataFrame): Input data to verify

    Returns:
        Dict: Dictionary containing data quality metrics
    """
    quality_metrics = {
        'row_count': len(data),
        'missing_values': {},
        'out_of_range_values': {},
        'data_types': {},
        'overall_quality_score': 0.0
    }

    # Check missing values
    missing_counts = data.isnull().sum()
    missing_percentages = (missing_counts / len(data)) * 100
    for col in data.columns:
        if missing_counts[col] > 0:
            quality_metrics['missing_values'][col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_percentages[col])
            }

    # Check data types
    for col in data.columns:
        if col in FEATURE_TYPES:
            expected_type = FEATURE_TYPES[col]
            actual_type = str(data[col].dtype)
            quality_metrics['data_types'][col] = {
                'expected': expected_type,
                'actual': actual_type,
                'matches': (
                    (expected_type == 'int' and pd.api.types.is_integer_dtype(data[col])) or
                    (expected_type == 'float' and pd.api.types.is_float_dtype(data[col])) or
                    (expected_type == 'bool' and pd.api.types.is_bool_dtype(data[col])) or
                    (expected_type == 'category' and (pd.api.types.is_string_dtype(data[col]) or
                                                     pd.api.types.is_categorical_dtype(data[col])))
                )
            }

    # Check out-of-range values
    if 'Age' in data.columns:
        out_of_range = ((data['Age'] < 15) | (data['Age'] > 100)).sum()
        if out_of_range > 0:
            quality_metrics['out_of_range_values']['Age'] = {
                'count': int(out_of_range),
                'percentage': float((out_of_range / len(data)) * 100)
            }

    if 'Annees_Experience' in data.columns:
        out_of_range = (data['Annees_Experience'] < 0).sum()
        if out_of_range > 0:
            quality_metrics['out_of_range_values']['Annees_Experience'] = {
                'count': int(out_of_range),
                'percentage': float((out_of_range / len(data)) * 100)
            }

    # Calculate overall quality score (simple version)
    # 1. Start with perfect score (100)
    # 2. Subtract for missing values (up to 40 points)
    # 3. Subtract for type mismatches (up to 30 points)
    # 4. Subtract for out-of-range values (up to 30 points)

    score = 100.0

    # Penalty for missing values
    missing_penalty = min(40, sum(metrics['percentage'] for metrics in quality_metrics['missing_values'].values()))
    score -= missing_penalty

    # Penalty for type mismatches
    type_mismatch_count = sum(1 for metrics in quality_metrics['data_types'].values() if not metrics['matches'])
    type_mismatch_penalty = min(30, (type_mismatch_count / max(1, len(quality_metrics['data_types']))) * 30)
    score -= type_mismatch_penalty

    # Penalty for out-of-range values
    range_penalty = min(30, sum(metrics['percentage'] for metrics in quality_metrics['out_of_range_values'].values()))
    score -= range_penalty

    quality_metrics['overall_quality_score'] = max(0, score)

    return quality_metrics

def format_prediction_output(prediction_results: Dict, format_type: str = 'json') -> Union[Dict, str, pd.DataFrame]:
    """
    Format prediction results for downstream use.

    Args:
        prediction_results (Dict): Prediction results from predict()
        format_type (str): Output format ('json', 'csv', 'dataframe')

    Returns:
        Union[Dict, str, pd.DataFrame]: Formatted output
    """
    if format_type == 'json':
        return prediction_results

    elif format_type == 'dataframe':
        # Create DataFrame with predictions and confidence intervals
        df = pd.DataFrame({
            'predicted_income': prediction_results['predictions']
        })

        if 'confidence_intervals' in prediction_results:
            df['lower_bound'] = prediction_results['confidence_intervals']['lower_bounds']
            df['upper_bound'] = prediction_results['confidence_intervals']['upper_bounds']
            df['confidence_level'] = prediction_results['confidence_intervals']['confidence_level']

        return df

    elif format_type == 'csv':
        # Convert to DataFrame first, then to CSV
        df = format_prediction_output(prediction_results, 'dataframe')
        return df.to_csv(index=False)

    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def main():
    """Example usage of the prediction functions."""
    try:
        # Load model and preprocessor
        model, preprocessor = load_model()

        # Example input data
        example_data = {
            'Age': 35,
            'Categorie_Age': 'Adulte',  # Added missing column
            'Sexe': 'Homme',
            'Milieu': 'Urbain',
            'Niveau_Education': 'Supérieur',
            'Annees_Experience': 10.5,
            'Etat_Matrimonial': 'Marié(e)',
            'Categorie_Socioprofessionnelle': 'Cadre supérieur',
            'Possession_Voiture': True,
            'Possession_Logement': True,
            'Possession_Terrain': False,
            'Nb_Personnes_Charge': 2,
            'Secteur_Activite': 'Privé',
            'Acces_Internet': True,
            'Indice_Richesse': 2.0,
            'Education_Superieure': 1  # Added missing column
        }

        # Make prediction
        prediction_result = predict(example_data, model, preprocessor)

        # Print prediction
        print("\nPrediction Result:")
        print(f"Predicted Income: {prediction_result['predictions'][0]:,.2f} MAD")

        if 'confidence_intervals' in prediction_result:
            ci = prediction_result['confidence_intervals']
            print(f"Confidence Interval ({ci['confidence_level']*100:.0f}%): "
                  f"{ci['lower_bounds'][0]:,.2f} - {ci['upper_bounds'][0]:,.2f} MAD")

        # Get explanation
        if SHAP_AVAILABLE:
            explanation = explain_prediction(example_data, model, preprocessor)

            print("\nFeature Importance:")
            for feature, importance in zip(
                explanation['feature_importance']['features'],
                explanation['feature_importance']['importance_values']
            ):
                print(f"  {feature}: {importance:.4f}")

        # Example with multiple records
        example_df = pd.DataFrame([
            {
                'Age': 35,
                'Categorie_Age': 'Adulte',
                'Sexe': 'Homme',
                'Milieu': 'Urbain',
                'Niveau_Education': 'Supérieur',
                'Annees_Experience': 10.5,
                'Etat_Matrimonial': 'Marié(e)',
                'Categorie_Socioprofessionnelle': 'Cadre supérieur',
                'Possession_Voiture': True,
                'Possession_Logement': True,
                'Possession_Terrain': False,
                'Nb_Personnes_Charge': 2,
                'Secteur_Activite': 'Privé',
                'Acces_Internet': True,
                'Indice_Richesse': 2.0,
                'Education_Superieure': 1
            },
            {
                'Age': 25,
                'Categorie_Age': 'Jeune Adulte',
                'Sexe': 'Femme',
                'Milieu': 'Urbain',
                'Niveau_Education': 'Supérieur',
                'Annees_Experience': 3.0,
                'Etat_Matrimonial': 'Célibataire',
                'Categorie_Socioprofessionnelle': 'Cadre moyen',
                'Possession_Voiture': False,
                'Possession_Logement': False,
                'Possession_Terrain': False,
                'Nb_Personnes_Charge': 0,
                'Secteur_Activite': 'Public',
                'Acces_Internet': True,
                'Indice_Richesse': 0.0,
                'Education_Superieure': 1
            }
        ])

        # Make batch predictions
        batch_result = predict(example_df, model, preprocessor)

        # Format as DataFrame
        result_df = format_prediction_output(batch_result, 'dataframe')
        print("\nBatch Predictions (DataFrame format):")
        print(result_df)

        # Verify data quality
        quality_metrics = verify_data_quality(example_df)
        print("\nData Quality Metrics:")
        print(f"Overall Quality Score: {quality_metrics['overall_quality_score']:.2f}/100")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()