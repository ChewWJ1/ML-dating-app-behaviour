import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
cache_file = os.path.join(root_dir, "models_v8", "tuned_results.joblib")

tuned_results = joblib.load(cache_file)
lgb_data = tuned_results['LightGBM']

print("LightGBM Tuned Keys:")
for k, v in lgb_data.items():
    if k != 'model' and k != 'y_pred' and k != 'y_prob':
        print(f"  {k}: {v}")
    elif k == 'model':
        print(f"  model: {type(v)}")
