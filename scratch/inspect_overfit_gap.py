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

def __getattr__(name):
    return MockClass
sys.modules['__main__'].__getattr__ = __getattr__

# Load baseline results
baseline = joblib.load('models_v5/baseline_results.joblib')

# Print Overfit Gap
print("Baseline Models Overfit Gap:")
for name, results in baseline.items():
    train_acc = results.get('train_acc')
    test_acc = results.get('test_acc')
    if train_acc is not None and test_acc is not None:
        gap = train_acc - test_acc
        print(f"Model: {name} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Overfit Gap: {gap:.4f}")
