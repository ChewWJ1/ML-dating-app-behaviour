import sys
from types import ModuleType

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

# Pre-populate base torch in sys.modules to satisfy top-level lookups
sys.modules['torch'] = MockModule('torch')

import joblib

# Catch-all module-level __getattr__ for __main__ module to handle all custom unpickling classes dynamically!
def __getattr__(name):
    return MockClass

# Put it in __main__
sys.modules['__main__'].__getattr__ = __getattr__

baseline = joblib.load('models_v5/baseline_results.joblib')
print("Baseline Models:")
for name, results in baseline.items():
    print(f"Model: {name}")
    print(f"  Test Acc: {results.get('test_acc')}")
    print(f"  F1: {results.get('f1')}")
    print(f"  ROC-AUC: {results.get('roc_auc')}")
