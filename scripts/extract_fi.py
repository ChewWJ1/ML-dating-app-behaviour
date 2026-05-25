import os
import joblib
import json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    results_path = os.path.join(base_dir, 'models', 'baseline_results.joblib')
    results = joblib.load(results_path)
    # print the keys
    print("Keys in baseline_results:", results.keys())
    
    # Try to extract feature importances from Random Forest or XGBoost
    importances = {}
    if 'Random Forest' in results and 'model' in results['Random Forest']:
        rf_model = results['Random Forest']['model']
        if hasattr(rf_model, 'feature_importances_'):
            importances['Random Forest'] = rf_model.feature_importances_.tolist()
    
    if 'XGBoost' in results and 'model' in results['XGBoost']:
        xgb_model = results['XGBoost']['model']
        if hasattr(xgb_model, 'feature_importances_'):
            importances['XGBoost'] = xgb_model.feature_importances_.tolist()
            
    # also print what keys are in each model's dict
    for k, v in results.items():
        print(f"Keys for {k}: {v.keys()}")
        if 'roc_auc' in v:
            print(f"  ROC AUC for {k}: {v['roc_auc']}")
        if 'accuracy' in v:
            print(f"  Accuracy for {k}: {v['accuracy']}")

    fi_path = os.path.join(base_dir, 'models', 'feature_importances.json')
    with open(fi_path, 'w') as f:
        json.dump(importances, f)
        
except Exception as e:
    print(f"Error: {e}")
