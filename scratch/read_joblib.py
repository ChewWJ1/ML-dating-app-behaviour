import joblib

path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\models\baseline_results.joblib"
results = joblib.load(path)

print("Keys of results:", list(results.keys()))
for name, r in results.items():
    print(f"\nModel: {name}")
    for k, v in r.items():
        if k not in ['model', 'y_pred', 'y_prob']:
            print(f"  {k}: {v}")
