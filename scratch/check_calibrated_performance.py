import sys
from types import ModuleType

# Mock torch completely to avoid loading DLLs
class MockClass:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return MockClass
    def __call__(self, *args, **kwargs):
        return MockClass()

class MockModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
    def __getattr__(self, name):
        return MockClass
    def __call__(self, *args, **kwargs):
        return MockClass()

class TorchImportHook:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('torch'):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        return MockModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, TorchImportHook())
sys.modules['torch'] = MockModule('torch')

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def __getattr__(name):
    return MockClass
sys.modules['__main__'].__getattr__ = __getattr__

# Load baseline results
baseline = joblib.load('models_v5/baseline_results.joblib')

# Load the actual train/test split data from baseline_results.joblib if stored, or reconstruct
# Wait, let's see what keys are stored in baseline_results.joblib
print("Keys in baseline_results:", list(baseline.keys()))

# Or let's see what is inside the 'Random Forest' baseline entry
rf_entry = baseline['Random Forest']
print("Keys in Random Forest entry:", list(rf_entry.keys()))
