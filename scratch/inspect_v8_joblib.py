import joblib
import sys
import os

sys.path.append(os.path.abspath('streamlit_app_v2'))
from utils.model_loader import inject_classes
inject_classes()

sys.stdout.reconfigure(encoding='utf-8')

baseline_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\models_v8\baseline_results.joblib"
tuned_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\models_v8\tuned_results.joblib"

if os.path.exists(baseline_path):
    print("Loading baseline results...")
    baseline_results = joblib.load(baseline_path)
    print("Baseline models in joblib:", list(baseline_results.keys()))
    for name, res in baseline_results.items():
        print(f"\nModel: {name}")
        for key in ['test_acc', 'f1', 'precision', 'recall', 'roc_auc']:
            val = res.get(key, 'N/A')
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
else:
    print("Baseline results not found!")

if os.path.exists(tuned_path):
    print("\n" + "="*40)
    print("Loading tuned results...")
    tuned_results = joblib.load(tuned_path)
    print("Tuned models in joblib:", list(tuned_results.keys()))
    for name, res in tuned_results.items():
        print(f"\nModel: {name}")
        for key in ['test_acc', 'f1', 'precision', 'recall', 'roc_auc', 'best_params']:
            val = res.get(key, 'N/A')
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
else:
    print("Tuned results not found!")
