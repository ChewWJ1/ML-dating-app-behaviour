import sys
import types
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Create stub modules for torch and its submodules to prevent loading c10.dll
stub_torch = types.ModuleType('torch')
stub_torch.nn = types.ModuleType('torch.nn')
stub_torch.nn.Module = object
stub_torch.utils = types.ModuleType('torch.utils')
stub_torch.utils.data = types.ModuleType('torch.utils.data')
stub_torch.utils.data.Dataset = object

# Add dummy device mock for torch.device
class DummyDevice:
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return "cpu"
    def __repr__(self):
        return "device(type='cpu')"

stub_torch.device = DummyDevice

# Register stubs in sys.modules
sys.modules['torch'] = stub_torch
sys.modules['torch.nn'] = stub_torch.nn
sys.modules['torch.utils'] = stub_torch.utils
sys.modules['torch.utils.data'] = stub_torch.utils.data

# 2. Stub the main module classes dynamically on request
class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name not in self.__dict__:
            # Check if it's PyTorchSklearnClassifier or other classifiers
            new_class = type(name, (BaseEstimator, ClassifierMixin), {})
            self.__dict__[name] = new_class
        return self.__dict__[name]

main = sys.modules['__main__']
stub_main = StubModule('__main__')
stub_main.__dict__.update(main.__dict__)
sys.modules['__main__'] = stub_main

import joblib
import os

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
cache_file = os.path.join(root_dir, "models_v8", "baseline_results.joblib")

if not os.path.exists(cache_file):
    print("Error: baseline_results.joblib does not exist!")
    sys.exit(1)

print("Attempting to load baseline results...")
try:
    results = joblib.load(cache_file)
    print("Loaded type:", type(results))
    if isinstance(results, dict):
        print(f"{'Model Name':<45} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'MCC':<10}")
        print("-" * 95)
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
                
                print(f"{name:<45} | {acc_str:<10} | {f1_str:<10} | {auc_str:<10} | {mcc_str:<10}")
            else:
                print(f"{name:<45} | {str(metrics)}")
except Exception as e:
    print("Failed to unpickle results:", e)
