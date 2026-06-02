import sys
import types
from sklearn.base import BaseEstimator, ClassifierMixin

# A simple dummy device class
class DummyDevice:
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return "cpu"
    def __repr__(self):
        return "device(type='cpu')"

class DummyClass:
    pass

# Define mock modules
def create_mock_module(name):
    mod = types.ModuleType(name)
    mod.__path__ = [] # treat as package
    return mod

# Create modules
torch = create_mock_module('torch')
torch.device = DummyDevice

torch_nn = create_mock_module('torch.nn')
torch_nn.Module = DummyClass
torch_nn.Parameter = DummyClass
torch_nn.ModuleList = DummyClass
torch_nn.Sequential = DummyClass

torch_nn_modules = create_mock_module('torch.nn.modules')

torch_nn_modules_module = create_mock_module('torch.nn.modules.module')
torch_nn_modules_module.Module = DummyClass

torch_nn_modules_container = create_mock_module('torch.nn.modules.container')
torch_nn_modules_container.ModuleList = DummyClass
torch_nn_modules_container.Sequential = DummyClass

torch_utils = create_mock_module('torch.utils')
torch_utils_data = create_mock_module('torch.utils.data')
torch_utils_data_dataset = create_mock_module('torch.utils.data.dataset')
torch_utils_data.Dataset = DummyClass

# Connect them
torch.nn = torch_nn
torch.nn.modules = torch_nn_modules
torch.nn.modules.module = torch_nn_modules_module
torch.nn.modules.container = torch_nn_modules_container
torch.utils = torch_utils
torch.utils.data = torch_utils_data
torch.utils.data.dataset = torch_utils_data_dataset

# Register in sys.modules
sys.modules['torch'] = torch
sys.modules['torch.device'] = DummyDevice
sys.modules['torch.nn'] = torch_nn
sys.modules['torch.nn.modules'] = torch_nn_modules
sys.modules['torch.nn.modules.module'] = torch_nn_modules_module
sys.modules['torch.nn.modules.container'] = torch_nn_modules_container
sys.modules['torch.nn.Parameter'] = DummyClass
sys.modules['torch.utils'] = torch_utils
sys.modules['torch.utils.data'] = torch_utils_data
sys.modules['torch.utils.data.dataset'] = torch_utils_data_dataset

# Stub the main module classes dynamically on request
class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name not in self.__dict__:
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
                
                acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
                auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
                mcc_str = f"{mcc:.4f}" if isinstance(mcc, float) else str(mcc)
                
                print(f"{name:<45} | {acc_str:<10} | {f1_str:<10} | {auc_str:<10} | {mcc_str:<10}")
except Exception as e:
    print("Failed to unpickle results:", e)
