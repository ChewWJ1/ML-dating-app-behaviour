import json
import os

# Paths
v5_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"
v4_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V4.ipynb"

# Full source code for V5
v5_bayesian_source = """import os, joblib
reports_dir = '../assets/notebook_plots' if os.path.exists('../assets') or not os.path.exists('assets') else 'assets/notebook_plots'
os.makedirs(reports_dir, exist_ok=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 1. Define model architecture
class BayesianMLP(nn.Module):
    \"\"\"MLP with dropout kept ON during inference for MC Dropout\"\"\"
    def __init__(self, n_features, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# 2. Cache Routing
os.makedirs('../models_v5', exist_ok=True)
cache_bayesian = '../models_v5/bayesian_model.joblib'

# Convert data safely to arrays
X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test

n_features = X_train_arr.shape[1]
bayesian_model = BayesianMLP(n_features=n_features).to(DEVICE)

if os.path.exists(cache_bayesian):
    print("⏭️  Loading pre-trained BayesianMLP model from cache...")
    state_dict = joblib.load(cache_bayesian)
    bayesian_model.load_state_dict(state_dict)
else:
    print("⏳ Training BayesianMLP model from scratch (~5-10s)...")
    X_train_t = torch.tensor(X_train_arr, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_arr, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(bayesian_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    bayesian_model.train()
    for epoch in range(15):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = bayesian_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    # Save model weights to cache (moving state dict parameters to CPU for safety)
    cpu_state_dict = {k: v.cpu() for k, v in bayesian_model.state_dict().items()}
    joblib.dump(cpu_state_dict, cache_bayesian)
    print("💾 BayesianMLP model weights cached successfully.")

# Train as normal, then use MC Dropout at inference
def mc_dropout_predict(model, X, n_forward=100):
    \"\"\"Run T stochastic forward passes to get predictive distribution\"\"\"
    model.train()  # KEEP dropout ON — this is the key trick
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.append(probs)
    
    predictions = np.array(predictions)  # Shape: (T, n_samples)
    
    mean_pred = predictions.mean(axis=0)        # Point estimate
    std_pred = predictions.std(axis=0)           # Epistemic uncertainty
    entropy = -(mean_pred * np.log(mean_pred + 1e-8) + 
                (1-mean_pred) * np.log(1-mean_pred + 1e-8))  # Predictive entropy
    
    return mean_pred, std_pred, entropy

mean_preds, uncertainties, entropies = mc_dropout_predict(bayesian_model, X_test_arr, n_forward=100)

# === REPORT VISUAL: Uncertainty analysis (3-panel) ===
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Uncertainty distribution
axes[0].hist(uncertainties, bins=50, color='#6c5ce7', alpha=0.7, edgecolor='white')
axes[0].axvline(uncertainties.mean(), color='red', linestyle='--', label=f'Mean: {uncertainties.mean():.3f}')
axes[0].set_xlabel('Predictive Uncertainty (σ)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Epistemic Uncertainty')
axes[0].legend()

# 2. Accuracy vs uncertainty — are confident predictions more accurate?
n_bins = 10
bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
bin_accs = []
bin_centers = []
for i in range(n_bins):
    mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
    if mask.sum() > 0:
        bin_preds = (mean_preds[mask] > 0.5).astype(int)
        bin_accs.append(accuracy_score(y_test.values[mask], bin_preds))
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

axes[1].bar(range(len(bin_accs)), bin_accs, color='#00b894', edgecolor='white')
axes[1].set_xlabel('Uncertainty Bin (Low → High)')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Uncertainty: Confident Predictions?')
axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)

# 3. Scatter: mean prediction vs uncertainty (colored by correctness)
correct = ((mean_preds > 0.5).astype(int) == y_test.values)
axes[2].scatter(mean_preds[correct], uncertainties[correct], alpha=0.15, s=5,
                color='#00b894', label='Correct')
axes[2].scatter(mean_preds[~correct], uncertainties[~correct], alpha=0.15, s=5,
                color='#d63031', label='Incorrect')
axes[2].set_xlabel('Mean Predicted Probability')
axes[2].set_ylabel('Uncertainty (σ)')
axes[2].set_title('Prediction Confidence Map')
axes[2].legend(markerscale=5)

plt.suptitle('Bayesian Uncertainty Quantification via Monte Carlo Dropout (T=100)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'bayesian_uncertainty.png'), dpi=150, bbox_inches='tight')
plt.show()

# Report table
print("\\n📊 Uncertainty-Aware Prediction Report:")
print(f"  High-confidence predictions (σ < median): {(uncertainties < np.median(uncertainties)).sum()}")
print(f"  Low-confidence predictions  (σ ≥ median): {(uncertainties >= np.median(uncertainties)).sum()}")
high_conf_acc = accuracy_score(y_test.values[uncertainties < np.median(uncertainties)],
                                (mean_preds[uncertainties < np.median(uncertainties)] > 0.5).astype(int))
low_conf_acc = accuracy_score(y_test.values[uncertainties >= np.median(uncertainties)],
                               (mean_preds[uncertainties >= np.median(uncertainties)] > 0.5).astype(int))
print(f"  High-confidence accuracy: {high_conf_acc:.4f}")
print(f"  Low-confidence accuracy:  {low_conf_acc:.4f}")
"""

# Update V5
with open(v5_path, 'r', encoding='utf-8') as f:
    nb_v5 = json.load(f)

for cell in nb_v5['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class BayesianMLP(nn.Module):" in source and "mc_dropout_predict(bayesian_model" in source:
            cell['source'] = [line + '\n' for line in v5_bayesian_source.split('\n')]
            print("Successfully injected trained & cached BayesianMC Dropout into V5!")
            break

with open(v5_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v5, f, indent=1, ensure_ascii=False)

# Update V4
v4_bayesian_source = v5_bayesian_source.replace('../models_v5', '../models_v4')

with open(v4_path, 'r', encoding='utf-8') as f:
    nb_v4 = json.load(f)

for cell in nb_v4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class BayesianMLP(nn.Module):" in source and "mc_dropout_predict(bayesian_model" in source:
            cell['source'] = [line + '\n' for line in v4_bayesian_source.split('\n')]
            print("Successfully injected trained & cached BayesianMC Dropout into V4!")
            break

with open(v4_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v4, f, indent=1, ensure_ascii=False)
