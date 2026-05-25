import os
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = joblib.load(os.path.join(base_dir, 'models', 'baseline_results.joblib'))
print(list(m['Random Forest']['model'].feature_names_in_))
