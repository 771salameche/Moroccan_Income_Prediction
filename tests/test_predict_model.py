#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Moroccan income prediction model.

This module tests the functionality of the predict_model.py script.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the module to test
from src.models import predict_model

class TestPredictModel(unittest.TestCase):
    """Test cases for the predict_model module."""

    def setUp(self):
        """Set up test fixtures."""
        # Example valid input data
        self.valid_data = {
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
        }

        # Example invalid input data (missing features)
        self.invalid_data_missing = {
            'Age': 35,
            'Sexe': 'Homme',
            # Missing other features
        }

        # Example invalid input data (invalid values)
        self.invalid_data_values = {
            'Age': 35,
            'Categorie_Age': 'Invalid',  # Invalid value
            'Sexe': 'Invalid',  # Invalid value
            'Milieu': 'Urbain',
            'Niveau_Education': 'Supérieur',
            'Annees_Experience': -5,  # Invalid negative value
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
        }

        # Example data with fixable issues
        self.fixable_data = {
            'Age': 35,
            'Categorie_Age': 'adulte',  # Lowercase, should be fixed
            'Sexe': 'homme',  # Lowercase, should be fixed
            'Milieu': 'Urbain',
            'Niveau_Education': 'Supérieur',
            'Annees_Experience': 10.5,
            'Etat_Matrimonial': 'Marié(e)',
            'Categorie_Socioprofessionnelle': 'Cadre supérieur',
            'Possession_Voiture': 'Oui',  # String instead of boolean
            'Possession_Logement': 1,  # Integer instead of boolean
            'Possession_Terrain': 0,  # Integer instead of boolean
            'Nb_Personnes_Charge': 2,
            'Secteur_Activite': 'Privé',
            'Acces_Internet': True,
            'Indice_Richesse': 2.0,
            'Education_Superieure': 1
        }

        # Try to load the model and preprocessor
        try:
            # Define OutlierCapper class to handle unpickling
            class OutlierCapper:
                def __init__(self, columns=None, factor=1.5):
                    self.columns = columns
                    self.factor = factor

                def transform(self, X):
                    return X

                def fit(self, X, y=None):
                    return self

                def fit_transform(self, X, y=None):
                    return self.transform(X)

            # Add the class to the global namespace for unpickling
            import sys
            sys.modules['OutlierCapper'] = OutlierCapper

            # Now try to load the model
            self.model, self.preprocessor = predict_model.load_model()
            self.model_loaded = True
        except (predict_model.ModelNotFoundError, predict_model.PipelineNotFoundError, AttributeError) as e:
            self.model_loaded = False
            print(f"Warning: Model or preprocessor not found or could not be loaded. Some tests will be skipped. Error: {str(e)}")

    def test_validate_input_data(self):
        """Test the validate_input_data function."""
        # Test with valid data
        is_valid, errors = predict_model.validate_input_data(pd.DataFrame([self.valid_data]))
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Test with missing features
        is_valid, errors = predict_model.validate_input_data(pd.DataFrame([self.invalid_data_missing]))
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

        # Test with invalid values
        is_valid, errors = predict_model.validate_input_data(pd.DataFrame([self.invalid_data_values]))
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_fix_common_issues(self):
        """Test the fix_common_issues function."""
        # Convert to DataFrame
        fixable_df = pd.DataFrame([self.fixable_data])

        # Fix issues
        fixed_df = predict_model.fix_common_issues(fixable_df)

        # Check if issues were fixed
        self.assertEqual(fixed_df['Sexe'].iloc[0], 'Homme')  # Should be capitalized
        self.assertEqual(fixed_df['Categorie_Age'].iloc[0], 'Adulte')  # Should be capitalized

        # Check if boolean values were converted correctly
        # The values might be True/False or 1/0 depending on how they're stored
        self.assertIn(fixed_df['Possession_Voiture'].iloc[0], [True, 1])
        self.assertIn(fixed_df['Possession_Logement'].iloc[0], [True, 1])
        self.assertIn(fixed_df['Possession_Terrain'].iloc[0], [False, 0])

    def test_predict(self):
        """Test the predict function."""
        if not self.model_loaded:
            self.skipTest("Model or preprocessor not found")

        # Test with valid data
        result = predict_model.predict(self.valid_data, self.model, self.preprocessor)

        # Check result structure
        self.assertIn('predictions', result)
        self.assertIn('input_shape', result)
        self.assertIn('prediction_count', result)
        self.assertIn('confidence_intervals', result)

        # Check prediction values
        self.assertEqual(len(result['predictions']), 1)
        self.assertGreater(result['predictions'][0], 0)  # Income should be positive

        # Check confidence intervals
        ci = result['confidence_intervals']
        self.assertIn('lower_bounds', ci)
        self.assertIn('upper_bounds', ci)
        self.assertIn('confidence_level', ci)
        self.assertEqual(len(ci['lower_bounds']), 1)
        self.assertEqual(len(ci['upper_bounds']), 1)
        self.assertGreaterEqual(ci['lower_bounds'][0], 0)  # Lower bound should be non-negative
        self.assertGreater(ci['upper_bounds'][0], ci['lower_bounds'][0])  # Upper bound should be greater than lower bound

        # Test with fixable data
        try:
            result = predict_model.predict(self.fixable_data, self.model, self.preprocessor)
            self.assertIn('predictions', result)  # Should work after fixing
        except Exception as e:
            self.fail(f"predict() raised {type(e).__name__} unexpectedly with fixable data: {str(e)}")

    def test_verify_data_quality(self):
        """Test the verify_data_quality function."""
        # Test with valid data
        quality_metrics = predict_model.verify_data_quality(pd.DataFrame([self.valid_data]))

        # Check metrics structure
        self.assertIn('row_count', quality_metrics)
        self.assertIn('missing_values', quality_metrics)
        self.assertIn('out_of_range_values', quality_metrics)
        self.assertIn('data_types', quality_metrics)
        self.assertIn('overall_quality_score', quality_metrics)

        # Check quality score
        self.assertGreaterEqual(quality_metrics['overall_quality_score'], 0)
        self.assertLessEqual(quality_metrics['overall_quality_score'], 100)

        # For valid data, score should be high
        self.assertGreater(quality_metrics['overall_quality_score'], 90)

    def test_format_prediction_output(self):
        """Test the format_prediction_output function."""
        # Create a sample prediction result
        prediction_result = {
            'predictions': [50000.0, 35000.0],
            'input_shape': (2, 14),
            'prediction_count': 2,
            'confidence_intervals': {
                'lower_bounds': [45000.0, 30000.0],
                'upper_bounds': [55000.0, 40000.0],
                'confidence_level': 0.95
            }
        }

        # Test JSON format (default)
        json_output = predict_model.format_prediction_output(prediction_result, 'json')
        self.assertEqual(json_output, prediction_result)

        # Test DataFrame format
        df_output = predict_model.format_prediction_output(prediction_result, 'dataframe')
        self.assertIsInstance(df_output, pd.DataFrame)
        self.assertEqual(len(df_output), 2)
        self.assertIn('predicted_income', df_output.columns)
        self.assertIn('lower_bound', df_output.columns)
        self.assertIn('upper_bound', df_output.columns)

        # Test CSV format
        csv_output = predict_model.format_prediction_output(prediction_result, 'csv')
        self.assertIsInstance(csv_output, str)
        self.assertIn('predicted_income', csv_output)
        self.assertIn('lower_bound', csv_output)
        self.assertIn('upper_bound', csv_output)

if __name__ == '__main__':
    unittest.main()
