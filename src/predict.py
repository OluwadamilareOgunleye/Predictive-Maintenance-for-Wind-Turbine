"""
Inference & Predictive Maintenance Diagnostics Module.
Provides functions to load models, score feature vectors, calculate Health Index,
and provide prescriptive maintenance recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_FILES = {
    'Random Forest': 'random_forest.joblib',
    'XGBoost': 'xgboost.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib',
    'Logistic Regression': 'logistic_regression.joblib',
    'K-Nearest Neighbors': 'k-nearest_neighbors.joblib'
}

SCALED_MODELS = {'Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors'}

_loaded_models = {}
_loaded_scaler = None
_feature_names = None

def get_feature_names():
    global _feature_names
    if _feature_names is None:
        path = os.path.join(MODELS_DIR, 'feature_names.json')
        with open(path, 'r') as f:
            _feature_names = json.load(f)
    return _feature_names

def load_scaler():
    global _loaded_scaler
    if _loaded_scaler is None:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        _loaded_scaler = joblib.load(scaler_path)
    return _loaded_scaler

def load_model(model_name='Random Forest'):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")
    if model_name not in _loaded_models:
        path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
        _loaded_models[model_name] = joblib.load(path)
    return _loaded_models[model_name]

def predict_single(features_dict, model_name='Random Forest'):
    """
    Predict health status for a single window feature set.
    
    Parameters:
        features_dict (dict): mapping of feature_name -> float value
        model_name (str): chosen classifier name
        
    Returns:
        dict containing:
            - status: 'Healthy' or 'Damaged'
            - damage_probability: float [0, 1]
            - health_index: float [0, 100]
            - severity: 'Optimal', 'Notice', 'Warning', 'Critical'
            - maintenance_action: string recommendation
    """
    feature_cols = get_feature_names()
    model = load_model(model_name)
    scaler = load_scaler()
    
    # Vectorize
    vector = np.array([float(features_dict.get(c, 0.0)) for c in feature_cols]).reshape(1, -1)
    
    if model_name in SCALED_MODELS:
        vector = scaler.transform(vector)
        
    # Prediction
    pred_label = int(model.predict(vector)[0])
    
    if hasattr(model, 'predict_proba'):
        prob_damage = float(model.predict_proba(vector)[0][1])
    else:
        prob_damage = 1.0 if pred_label == 1 else 0.0
        
    health_index = max(0.0, min(100.0, (1.0 - prob_damage) * 100.0))
    status = 'Healthy' if pred_label == 0 else 'Damaged'
    
    if health_index >= 85:
        severity = 'Optimal'
        action = 'Turbine operating normally. Next routine maintenance in 180 days.'
    elif health_index >= 60:
        severity = 'Notice'
        action = 'Minor vibration anomaly detected. Schedule inspection during next planned outage.'
    elif health_index >= 30:
        severity = 'Warning'
        action = 'Elevated drivetrain vibration detected. Verify lubrication and monitor high-speed shaft bearing.'
    else:
        severity = 'Critical'
        action = 'URGENT: Severe vibration defect signature! Reduce turbine generation load and dispatch maintenance crew immediately.'

    return {
        'status': status,
        'damage_probability': round(prob_damage * 100, 1),
        'health_index': round(health_index, 1),
        'severity': severity,
        'maintenance_action': action,
        'model_used': model_name
    }
