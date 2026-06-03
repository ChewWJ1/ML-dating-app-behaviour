import json

notebook_path = 'notebooks/ML_dating_app_behaviour V8_patched.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell(query):
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if query in source:
                return cell
    return None

# 1. Causal Pre-Split Scaling Fix
cell_causal = find_cell("scaler_temp = RobustScaler()")
if cell_causal:
    source = "".join(cell_causal['source'])
    old_causal = """scaler_temp = RobustScaler()
df_temp[numeric_cols] = scaler_temp.fit_transform(df_temp[numeric_cols])
df_processed = df_temp
X = df_processed.drop(columns=['target'])
y = df_processed['target']"""
    new_causal = """from sklearn.model_selection import train_test_split
X_temp_train, X_temp_test = train_test_split(df_temp, test_size=0.2, random_state=RANDOM_STATE)
scaler_temp = RobustScaler()
X_temp_train[numeric_cols] = scaler_temp.fit_transform(X_temp_train[numeric_cols])
df_processed = X_temp_train
X = df_processed.drop(columns=['target'])
y = df_processed['target']"""
    source = source.replace(old_causal, new_causal)
    cell_causal['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 2. Generate Real Regularization Loss Curves
cell_sim = find_cell("Simulate comparative loss curves showing the smoothing effect of mixup")
if cell_sim:
    new_sim_source = """# --- V5.1 COMPARATIVE VALIDATION: LABEL SMOOTHING & MIXUP ---
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("⏳ Training SimpleMLP to generate real loss curves...")

class SimpleMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Split a small validation set
from sklearn.model_selection import train_test_split
# Ensure DEVICE is available, fallback to CPU
local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if hasattr(X_train, "values"):
    X_t_arr, y_t_arr = X_train.values, y_train.values
else:
    X_t_arr, y_t_arr = X_train, y_train

X_t, X_v, y_t, y_v = train_test_split(X_t_arr, y_t_arr, test_size=0.2, random_state=RANDOM_STATE)
X_t_tens = torch.tensor(X_t, dtype=torch.float32).to(local_device)
y_t_tens = torch.tensor(y_t, dtype=torch.float32).to(local_device)
X_v_tens = torch.tensor(X_v, dtype=torch.float32).to(local_device)
y_v_tens = torch.tensor(y_v, dtype=torch.float32).to(local_device)

loader = DataLoader(TensorDataset(X_t_tens, y_t_tens), batch_size=512, shuffle=True)
criterion = nn.BCEWithLogitsLoss()

def train_eval(model, optimizer, use_mixup=False):
    train_losses, val_losses = [], []
    for epoch in range(15):
        model.train()
        epoch_loss = 0.0
        for b_X, b_y in loader:
            optimizer.zero_grad()
            if use_mixup:
                # Label Smoothing
                b_y_smooth = b_y * 0.8 + 0.1
                # Mixup
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(b_X.size(0)).to(local_device)
                mixed_X = lam * b_X + (1 - lam) * b_X[idx]
                mixed_y = lam * b_y_smooth + (1 - lam) * b_y_smooth[idx]
                preds = model(mixed_X)
                loss = criterion(preds, mixed_y)
            else:
                preds = model(b_X)
                loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(loader))
        model.eval()
        with torch.no_grad():
            v_preds = model(X_v_tens)
            v_loss = criterion(v_preds, y_v_tens)
            val_losses.append(v_loss.item())
    return train_losses, val_losses

# Train Standard Model
model_std = SimpleMLP(X_t.shape[1]).to(local_device)
opt_std = optim.Adam(model_std.parameters(), lr=0.01)
std_train_loss, std_test_loss = train_eval(model_std, opt_std, use_mixup=False)

# Train Mixup Model
model_mix = SimpleMLP(X_t.shape[1]).to(local_device)
opt_mix = optim.Adam(model_mix.parameters(), lr=0.01)
mix_train_loss, mix_test_loss = train_eval(model_mix, opt_mix, use_mixup=True)

epochs_range = np.arange(1, 16)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, std_train_loss, 'o-', color='#e91e63', label='Train (Standard BCE)')
plt.plot(epochs_range, std_test_loss, 's--', color='#f48fb1', label='Test (Standard BCE)')
plt.title('Standard Training (Real Data)')
plt.xlabel('Epochs')
plt.ylabel('BCE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, mix_train_loss, 'o-', color='#2196f3', label='Train (Smoothed Mixup)')
plt.plot(epochs_range, mix_test_loss, 's--', color='#90caf9', label='Test (Smoothed Mixup)')
plt.title('Label Smoothing & Tabular Mixup (Real Data)')
plt.xlabel('Epochs')
plt.ylabel('Regularized BCE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('🛡️ Regularization Analysis: Real Empirical Validation on Training Set', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'regularization_real_curves.png') if 'reports_dir' in globals() else 'regularization_real_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""
    cell_sim['source'] = [line + ('\n' if i < len(new_sim_source.split('\n')) - 1 else '') for i, line in enumerate(new_sim_source.split('\n'))]

with open('notebooks/ML_dating_app_behaviour V8_patched_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Phase 2 patches applied successfully!")
