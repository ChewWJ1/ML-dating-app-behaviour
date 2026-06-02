import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
cache_file = os.path.join(root_dir, "models_v8", "tuned_results.joblib")

if not os.path.exists(cache_file):
    print("Error: tuned_results.joblib does not exist!")
    sys.exit(1)

tuned_results = joblib.load(cache_file)
print("Loaded type:", type(tuned_results))

for name, data in tuned_results.items():
    print(f"\n==========================================")
    print(f"MODEL: {name}")
    print(f"==========================================")
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ['best_params', 'best_params_', 'best_cv_score', 'test_acc', 'precision', 'recall', 'f1', 'roc_auc', 'tune_time']:
                print(f"  {k}: {v}")
    else:
        print(f"  Data: {data}")
