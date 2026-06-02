import joblib
import sys
import os

sys.path.append(os.path.abspath('streamlit_app_v2'))
try:
    from utils.model_loader import inject_classes
    inject_classes()
except Exception as e:
    print("Could not inject classes:", e)

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
models_dir = os.path.join(root_dir, "models_v8")

files_to_inspect = [
    "opacus.joblib",
    "scarf.joblib",
    "gnn_gat.joblib",
    "tabpfn.joblib",
    "causal_uplift.joblib",
    "flaml_results.joblib",
    "pycaret_results.joblib",
    "mapie.joblib",
    "dml_causal.joblib",
    "bayesian_model.joblib"
]

for filename in files_to_inspect:
    path = os.path.join(models_dir, filename)
    print("\n" + "="*60)
    print(f"Inspecting file: {filename}")
    print("="*60)
    if not os.path.exists(path):
        print("File does not exist!")
        continue
    try:
        data = joblib.load(path)
        print("Type:", type(data))
        if isinstance(data, dict):
            print("Keys in dictionary:", list(data.keys()))
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"  {k}: {v}")
                elif k in ['metrics', 'test_metrics', 'results', 'scores', 'test_results']:
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  Sub-keys in {k}: {list(v.keys())}")
                    # Print small values
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float, str, bool)):
                            print(f"    {sk}: {sv}")
        else:
            print("Loaded object:", repr(data)[:300])
            # If it has attributes like 'metrics_' or similar, print them
            for attr in dir(data):
                if attr.endswith('_') and not attr.startswith('_'):
                    try:
                        val = getattr(data, attr)
                        if isinstance(val, (int, float, str, bool, dict, list)):
                            print(f"  Attribute {attr}: {repr(val)[:200]}")
                    except:
                        pass
    except Exception as e:
        print("Failed to load or inspect:", e)
