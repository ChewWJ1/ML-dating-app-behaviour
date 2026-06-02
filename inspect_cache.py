import joblib
import sys
import os

sys.path.append(os.path.abspath('streamlit_app_v2'))
from utils.model_loader import inject_classes
inject_classes()

def inspect(path):
    print(f"Inspecting {path}...")
    try:
        data = joblib.load(path)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"Keys: {keys}")
            if len(keys) > 0:
                first_key = keys[0]
                first_val = data[first_key]
                print(f"Key ({first_key}) type: {type(first_val)}")
                if isinstance(first_val, dict):
                    print(f"First key inner keys: {list(first_val.keys())}")
    except Exception as e:
        print(f"Error: {e}")

inspect('models_v8/tuned_results.joblib')
inspect('models_v8/pycaret_results.joblib')
inspect('models_v8/flaml_results.joblib')
