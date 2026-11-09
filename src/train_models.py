"""
Model Training and Benchmark Suite for Wind Turbine Predictive Maintenance.
Trains and compares multiple machine learning models:
1. Random Forest Classifier
2. XGBoost Classifier
3. Support Vector Classifier (RBF Kernel)
4. Logistic Regression
5. K-Nearest Neighbors Classifier

Evaluates on out-of-session holdout test turbines and exports trained artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load processed splits
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_features.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_features.csv'))
    
    # Feature columns (exclude metadata columns)
    meta_cols = ['file_id', 'window_idx', 'label', 'status']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"Features count: {len(feature_cols)}")
    print(f"Train samples: {X_train.shape[0]} (Healthy: {np.sum(y_train==0)}, Damaged: {np.sum(y_train==1)})")
    print(f"Test samples:  {X_test.shape[0]} (Healthy: {np.sum(y_test==0)}, Damaged: {np.sum(y_test==1)})")
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
        
    models_dict = {
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
            False  # Tree-based does not require scaling, but works with raw or scaled
        ),
        'XGBoost': (
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='logloss'),
            False
        ),
        'Support Vector Machine': (
            SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
            True   # Needs scaled features
        ),
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            True   # Needs scaled features
        ),
        'K-Nearest Neighbors': (
            KNeighborsClassifier(n_neighbors=5, weights='distance'),
            True   # Needs scaled features
        )
    }
    
    results = {}
    evaluation_details = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, (clf, use_scaled) in models_dict.items():
        print(f"\n---> Training: {name}...")
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Train on full train set
        clf.fit(X_tr, y_train)
        
        # Predictions & probabilities
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # downsample roc curve points for clean JSON storage
        indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
        
        results[name] = {
            'cv_accuracy_mean': round(cv_mean * 100, 2),
            'cv_accuracy_std': round(cv_std * 100, 2),
            'test_accuracy': round(acc * 100, 2),
            'test_precision': round(prec * 100, 2),
            'test_recall': round(rec * 100, 2),
            'test_f1': round(f1 * 100, 2),
            'test_roc_auc': round(auc * 100, 2),
            'uses_scaled_input': use_scaled
        }
        
        evaluation_details[name] = {
            'confusion_matrix': cm,
            'roc_curve': {
                'fpr': [float(x) for x in fpr[indices]],
                'tpr': [float(x) for x in tpr[indices]]
            }
        }
        
        # Save model
        clean_name = name.lower().replace(' ', '_')
        model_save_path = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
        joblib.dump(clf, model_save_path)
        print(f"[{name}] Test Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}%")
    
    # Feature importances from Random Forest & XGBoost
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_importances,
        'xgb_importance': xgb_importances
    })
    importance_df['mean_importance'] = (importance_df['rf_importance'] + importance_df['xgb_importance']) / 2.0
    importance_df = importance_df.sort_values(by='mean_importance', ascending=False)
    
    top_features = importance_df.head(20).to_dict(orient='records')
    
    # Channel importance breakdown
    channel_imp = {}
    for ch in ['AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']:
        ch_rows = importance_df[importance_df['feature'].str.startswith(f"{ch}_")]
        channel_imp[ch] = float(ch_rows['mean_importance'].sum())
        
    with open(os.path.join(MODELS_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, 'evaluation_details.json'), 'w') as f:
        json.dump(evaluation_details, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, 'feature_importances.json'), 'w') as f:
        json.dump({
            'top_features': top_features,
            'channel_breakdown': channel_imp
        }, f, indent=2)
        
    print("\nAll models trained and saved successfully in:", MODELS_DIR)
    print("Summary of Results:")
    summary_df = pd.DataFrame(results).T
    print(summary_df[['cv_accuracy_mean', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']])

if __name__ == '__main__':
    train_and_evaluate()
