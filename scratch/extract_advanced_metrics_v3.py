import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

# Import real torch and patch it
try:
    import torch
    import torch.utils.data
    # Patch Sampler class to be subscriptable in python 3.9+
    torch.utils.data.Sampler.__class_getitem__ = classmethod(lambda cls, item: cls)
    print("Successfully patched torch.utils.data.Sampler")
except Exception as e:
    print("Could not import or patch torch:", e)

# Import real opacus if possible and patch its classes if needed
try:
    import opacus
    print("Successfully imported opacus")
except Exception as e:
    print("Could not import opacus:", e)

# Load dataset and get y_test
root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
csv_path = os.path.join(root_dir, "data", "dating_app_behavior_dataset_extended1.csv")

print("Loading dataset...")
df = pd.read_csv(csv_path)
positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
y = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)

# Duplicate features placeholder
X = df.drop(columns=['match_outcome'])

# Train/Test Split (deterministic)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load opacus.joblib
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

# Load tabpfn.joblib
print("\n=== TABPFN MODEL ===")
tabpfn_path = os.path.join(root_dir, "models_v8", "tabpfn.joblib")
if os.path.exists(tabpfn_path):
    try:
        data = joblib.load(tabpfn_path)
        print("TabPFN keys:", list(data.keys()))
        pred = data['pred']
        print(f"Pred shape: {pred.shape}")
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, pred)
        
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"ROC-AUC:   {auc:.4f}")
    except Exception as e:
        print("Failed to inspect TabPFN:", e)

# Load GAT
print("\n=== GRAPH ATTENTION NETWORK (GAT) ===")
gat_path = os.path.join(root_dir, "models_v8", "gnn_gat.joblib")
if os.path.exists(gat_path):
    try:
        data = joblib.load(gat_path)
        print("GAT keys:", list(data.keys()))
        print(f"GAT Test Accuracy: {data.get('test_acc', 'N/A')}")
    except Exception as e:
        print("Failed to load GAT:", e)
