import sys
import os
import types
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

# Mock classes to support subscriptability
class DummyClass(object):
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return DummyClass
    def __call__(self, *args, **kwargs):
        return self
    def __class_getitem__(cls, item):
        return cls

class DummyDevice(object):
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return "cpu"
    def __repr__(self):
        return "device(type='cpu')"
    def __class_getitem__(cls, item):
        return cls

# Install finder to mock torch imports and submodules
class TorchFinder(object):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('torch') or fullname.startswith('opacus'):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None
        
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = [] # treat as package
        
        # Populate common classes/attributes
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
        name = module.__name__
        parts = name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            if parent_name in sys.modules:
                parent = sys.modules[parent_name]
                setattr(parent, parts[-1], module)

sys.meta_path.insert(0, TorchFinder())

sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch'].device = DummyDevice
sys.modules['torch.utils'] = types.ModuleType('torch.utils')
sys.modules['torch.utils.data'] = types.ModuleType('torch.utils.data')
sys.modules['torch.utils.data'].Sampler = DummyClass
sys.modules['torch.utils.data'].Dataset = DummyClass
sys.modules['torch.utils.data'].DataLoader = DummyClass

# Load data and get y_test
root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
csv_path = os.path.join(root_dir, "data", "dating_app_behavior_dataset_extended1.csv")

print("Loading dataset...")
df = pd.read_csv(csv_path)
positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
y = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)

# Duplicate the features placeholder
X = df.drop(columns=['match_outcome']) # placeholder for split

# Train/Test Split (deterministic)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("y_test class balance:")
print(y_test.value_counts())

# Now inspect opacus.joblib
print("\n=== OPACUS DP-SGD MODEL ===")
opacus_path = os.path.join(root_dir, "models_v8", "opacus.joblib")
if os.path.exists(opacus_path):
    try:
        data = joblib.load(opacus_path)
        print("Opacus keys:", list(data.keys()))
        y_pred = data['y_pred']
        print(f"y_pred shape: {y_pred.shape}")
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred)
        
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Epsilon:   {data.get('epsilon', 'N/A')}")
    except Exception as e:
        print("Failed to inspect Opacus:", e)

# Inspect tabpfn.joblib
print("\n=== TABPFN MODEL ===")
tabpfn_path = os.path.join(root_dir, "models_v8", "tabpfn.joblib")
if os.path.exists(tabpfn_path):
    try:
        data = joblib.load(tabpfn_path)
        print("TabPFN keys:", list(data.keys()))
        pred = data['pred']
        print(f"Pred shape: {pred.shape}")
        
        # In the notebook, TabPFN is evaluated on the test subset (first 1000 samples of X_test)
        y_test_subset = y_test[:len(pred)]
        acc = accuracy_score(y_test_subset, pred)
        f1 = f1_score(y_test_subset, pred, zero_division=0)
        prec = precision_score(y_test_subset, pred, zero_division=0)
        rec = recall_score(y_test_subset, pred, zero_division=0)
        auc = roc_auc_score(y_test_subset, pred)
        
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"ROC-AUC:   {auc:.4f}")
    except Exception as e:
        print("Failed to inspect TabPFN:", e)

# Inspect GAT
print("\n=== GRAPH ATTENTION NETWORK (GAT) ===")
gat_path = os.path.join(root_dir, "models_v8", "gnn_gat.joblib")
if os.path.exists(gat_path):
    try:
        data = joblib.load(gat_path)
        print("GAT keys:", list(data.keys()))
        print(f"GAT Test Accuracy: {data.get('test_acc', 'N/A')}")
    except Exception as e:
        print("Failed to load GAT:", e)
