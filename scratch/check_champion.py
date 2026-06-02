import joblib

def main():
    try:
        tuned = joblib.load(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\models_v8\tuned_results.joblib")
        print("TUNED RESULTS KEYS:", tuned.keys())
        for model_name, data in tuned.items():
            print(f"Model: {model_name}")
            print(f"  ROC-AUC: {data.get('roc_auc')}")
            print(f"  Accuracy: {data.get('accuracy')}")
            print(f"  F1: {data.get('f1')}")
    except Exception as e:
        print("Error loading tuned_results:", e)

if __name__ == "__main__":
    main()
