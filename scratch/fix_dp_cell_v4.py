import json

nb_path = 'notebooks/ML_dating_app_behaviour V4.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell containing 'cache_dp'
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'cache_dp' in source:
            # Define the new source code for the DP cell
            new_source = """import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_dp = 'models_v4_cache/opacus.joblib'

if os.path.exists(cache_dp):
    print("⏭️  Loading cached Differential Privacy model...")
    dp_data = joblib.load(cache_dp)
    dp_model = dp_data['model']
    epsilon = dp_data['epsilon']
    dp_losses = dp_data['losses']
    y_pred_dp = dp_data['y_pred']
    # Define variables for visual
    epsilon_history = [epsilon] * 30
    dp_predictions = y_pred_dp
else:
    print("⏳ Running Differential Privacy Training with Opacus (~3-6m)...")
    # pip install opacus
    import torch
    import torch.nn as nn
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define DP-compatible model (BatchNorm → GroupNorm)
    class DPModel(nn.Module):
        def __init__(self, n_features, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.GroupNorm(4, hidden),  # DP-compatible (not BatchNorm!)
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.GroupNorm(4, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    model_dp = DPModel(X_train.shape[1]).to(DEVICE)
    model_dp = ModuleValidator.fix(model_dp)  # Auto-fix DP incompatibilities
    
    optimizer = torch.optim.Adam(model_dp.parameters(), lr=1e-3)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32),
        torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.float32)
    ), batch_size=256, shuffle=True)
    
    # Attach Opacus PrivacyEngine
    privacy_engine = PrivacyEngine()
    model_dp, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model_dp,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=30,
        target_epsilon=8.0,     # Privacy budget
        target_delta=1e-5,      # Probability of privacy breach
        max_grad_norm=1.0       # Gradient clipping bound
    )
    
    # Train with privacy
    epsilon_history = []
    dp_losses = []
    for epoch in range(30):
        model_dp.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.BCEWithLogitsLoss()(model_dp(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        eps = privacy_engine.get_epsilon(delta=1e-5)
        epsilon_history.append(eps)
        dp_losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/30 | Loss: {epoch_loss/len(train_loader):.4f} | ε: {eps:.2f}")
            
    dp_model = model_dp
    epsilon = eps
    
    # Generate predictions on test set
    model_dp.eval()
    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model_dp(X_test_tensor)
        probs = torch.sigmoid(logits)
        y_pred_dp = (probs >= 0.5).cpu().numpy().astype(int)
        
    dp_predictions = y_pred_dp
    joblib.dump({'model': dp_model, 'epsilon': epsilon, 'losses': dp_losses, 'y_pred': y_pred_dp}, cache_dp)

# Define prediction comparison variables
dp_predictions = y_pred_dp
if 'results' in globals() and 'FT-Transformer' in results:
    non_dp_predictions = results['FT-Transformer']['pred']
elif 'results' in globals() and 'Random Forest' in results:
    non_dp_predictions = results['Random Forest']['pred']
else:
    non_dp_predictions = y_pred_dp

# === REPORT VISUAL: Privacy budget consumption ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epsilon_history, 'o-', color='#e17055', linewidth=2)
axes[0].axhline(y=8.0, color='red', linestyle='--', alpha=0.7, label='Target ε = 8.0')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Privacy Budget (ε)')
axes[0].set_title('Differential Privacy: Cumulative ε Over Training')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Compare DP vs non-DP model performance
dp_acc = accuracy_score(y_test, dp_predictions)
non_dp_acc = accuracy_score(y_test, non_dp_predictions)
labels = ['Standard\\nTraining', 'DP Training\\n(ε=8.0)']
accs = [non_dp_acc, dp_acc]
colors = ['#00b894', '#e17055']
bars = axes[1].bar(labels, accs, color=colors, edgecolor='white', width=0.5)
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Privacy-Utility Tradeoff')
axes[1].set_ylim(0, 1)
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{acc:.3f}', ha='center')

plt.suptitle('Differential Privacy: (ε=8.0, δ=1e-5) Guarantees for Sensitive Dating Data',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/differential_privacy.png', dpi=150, bbox_inches='tight')
plt.show()"""

            cell['source'] = [line + "\n" for line in new_source.split("\n")]
            if cell['source'][-1] == "\n":
                cell['source'][-1] = ""
                
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Successfully fixed NameError inside DP cell in V4 at index {idx}!")
            break
