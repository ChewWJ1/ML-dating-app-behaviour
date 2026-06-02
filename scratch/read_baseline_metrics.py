import joblib
import os
import sys
from sklearn.base import BaseEstimator, ClassifierMixin

# Stub the custom class to allow joblib to unpickle it
class PyTorchSklearnClassifier(BaseEstimator, ClassifierMixin):
    pass

# Inject it into __main__ so joblib finds it
import __main__
__main__.PyTorchSklearnClassifier = PyTorchSklearnClassifier

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
cache_file = os.path.join(root_dir, "models_v8", "baseline_results.joblib")

if not os.path.exists(cache_file):
    print("Error: baseline_results.joblib does not exist!")
    sys.exit(1)

results = joblib.load(cache_file)
print("Loaded type:", type(results))

if isinstance(results, dict):
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'MCC':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        if isinstance(metrics, dict):
            acc = metrics.get('test_acc', metrics.get('accuracy', 'N/A'))
            f1 = metrics.get('test_f1', metrics.get('f1', 'N/A'))
            auc = metrics.get('test_auc', metrics.get('roc_auc', 'N/A'))
            mcc = metrics.get('test_mcc', metrics.get('mcc', 'N/A'))
            
            # Format nicely
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            mcc_str = f"{mcc:.4f}" if isinstance(mcc, float) else str(mcc)
            
            print(f"{name:<30} | {acc_str:<10} | {f1_str:<10} | {auc_str:<10} | {mcc_str:<10}")
        else:
            print(f"{name:<30} | {str(metrics)}")
else:
    print(results)
