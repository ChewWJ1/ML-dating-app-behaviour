import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
cache_dir = os.path.join(root_dir, "models_v8")

def check_cache_file(name):
    path = os.path.join(cache_dir, name)
    print(f"\nChecking cache file: {name}")
    if not os.path.exists(path):
        print("Does not exist!")
        return None
    try:
        data = joblib.load(path)
        print(f"Loaded type: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", list(data.keys()))
            for k in list(data.keys())[:5]:
                print(f"  {k}: {type(data[k])}")
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

results = check_cache_file("baseline_results.joblib")
if results and isinstance(results, dict):
    # Print metrics for each model
    print("\nBaseline Model Metrics:")
    for model_name, metrics in results.items():
        if isinstance(metrics, dict):
            # Print keys inside metrics
            print(f"  {model_name}:")
            for m_key, val in metrics.items():
                if m_key in ['test_acc', 'test_f1', 'test_auc', 'test_mcc', 'accuracy', 'f1', 'roc_auc', 'mcc']:
                    print(f"    {m_key}: {val}")
        else:
            print(f"  {model_name}: {metrics}")

tuned_results = check_cache_file("tuned_results.joblib")
if tuned_results and isinstance(tuned_results, dict):
    print("\nTuned Model Metrics:")
    for model_name, metrics in tuned_results.items():
        if isinstance(metrics, dict):
            print(f"  {model_name}:")
            for m_key, val in metrics.items():
                if m_key in ['test_acc', 'test_f1', 'test_auc', 'test_mcc', 'accuracy', 'f1', 'roc_auc', 'mcc']:
                    print(f"    {m_key}: {val}")
        else:
            print(f"  {model_name}: {metrics}")
