import sys
import types
from sklearn.base import BaseEstimator, ClassifierMixin

sys.stdout.reconfigure(encoding='utf-8')

# A simple dummy device class
class DummyDevice:
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return "cpu"
    def __repr__(self):
        return "device(type='cpu')"

class DummyClass:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return DummyClass
    def __call__(self, *args, **kwargs):
        return self

# Custom import finder to dynamically resolve torch imports
class TorchFinder(object):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('torch') or fullname.startswith('opacus'):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None
        
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = [] # treat as package
        
        # Populate common classes/attributes on dynamically created module
        mod.device = DummyDevice
        mod.Module = DummyClass
        mod.Parameter = DummyClass
        mod.ModuleList = DummyClass
        mod.Sequential = DummyClass
        mod.Linear = DummyClass
        mod.BCEWithLogitsLoss = DummyClass
        mod.Sigmoid = DummyClass
        mod.ReLU = DummyClass
        mod.Dropout = DummyClass
        mod.BatchNorm1d = DummyClass
        mod.Dataset = DummyClass
        mod.DataLoader = DummyClass
        
        return mod
        
    def exec_module(self, module):
        # We can also dynamically attach submodules to parent packages
        name = module.__name__
        parts = name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            if parent_name in sys.modules:
                parent = sys.modules[parent_name]
                setattr(parent, parts[-1], module)

# Install finder
sys.meta_path.insert(0, TorchFinder())

# Populate basic entry points in sys.modules to satisfy import lookups
sys.modules['torch'] = sys.modules['torch'] if 'torch' in sys.modules else types.ModuleType('torch')
sys.modules['torch.nn'] = sys.modules['torch.nn'] if 'torch.nn' in sys.modules else types.ModuleType('torch.nn')
sys.modules['torch.utils'] = sys.modules['torch.utils'] if 'torch.utils' in sys.modules else types.ModuleType('torch.utils')

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
