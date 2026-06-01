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

sys.modules['torch'] = MockModule('torch')

import joblib

def __getattr__(name):
    return MockClass

sys.modules['__main__'].__getattr__ = __getattr__

# Load original baseline
try:
    orig_baseline = joblib.load('models/baseline_results.joblib')
    print("Original Baseline Models:")
    for name, results in orig_baseline.items():
        print(f"Model: {name} | Test Acc: {results.get('test_acc')} | F1: {results.get('f1')}")
except Exception as e:
    print(f"Error loading original baseline: {e}")

# Load original tuned
try:
    orig_tuned = joblib.load('models/tuned_results.joblib')
    print("\nOriginal Tuned Models:")
    for name, results in orig_tuned.items():
        print(f"Model: {name} | Test Acc: {results.get('test_acc')} | F1: {results.get('f1')}")
except Exception as e:
    print(f"Error loading original tuned: {e}")
