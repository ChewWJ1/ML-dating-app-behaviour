import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

# Load the notebook JSON
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Updating SCARF cell (Cell 121) to optimize t-SNE visualization footprint...")

scarf_source = """import os
reports_dir = '../assets/notebook_plots' if os.path.exists('../assets') or not os.path.exists('assets') else 'assets/notebook_plots'
os.makedirs(reports_dir, exist_ok=True)
import os, joblib
os.makedirs('../models_v5', exist_ok=True)
cache_scarf = '../models_v5/scarf.joblib'

# Validate that preceding split variables are defined in the interactive session
if 'y_train' not in globals() or 'X_train' not in globals() or 'X_test' not in globals() or 'y_test' not in globals():
    raise NameError("❌ Required variables (X_train, y_train, X_test, y_test) are not defined in the active session.\\n"
                    "👉 Please run the preceding cells first (specifically Cell 78: 'Train / Test Split') to populate these variables in memory.")

if os.path.exists(cache_scarf):
    print("⏭️  Loading cached SCARF representations...")
    scarf_data = joblib.load(cache_scarf)
    X_train_embed = scarf_data['X_train_embed']
    X_test_embed = scarf_data['X_test_embed']
    pretrain_losses = scarf_data.get('pretrain_losses', None)
else:
    print("⏳ Running SCARF Contrastive Pre-Training (~2-5m)...")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ScarfEncoder(nn.Module):
        \"\"\"Encoder network for SCARF contrastive learning\"\"\"
        def __init__(self, n_features, d_hidden=128, d_embed=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, d_hidden),
                nn.BatchNorm1d(d_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_hidden, d_hidden),
                nn.BatchNorm1d(d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_embed)
            )
        def forward(self, x):
            return self.encoder(x)
    
    class ScarfProjector(nn.Module):
        \"\"\"Projection head (discarded after pre-training)\"\"\"
        def __init__(self, d_embed=64, d_proj=32):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(d_embed, d_proj),
                nn.ReLU(),
                nn.Linear(d_proj, d_proj)
            )
        def forward(self, x):
            return self.proj(x)
    
    def scarf_corrupt(X_batch, X_all, corruption_rate=0.6):
        \"\"\"Corrupt features by replacing with random values from marginal distributions\"\"\"
        mask = torch.bernoulli(torch.full_like(X_batch, corruption_rate))
        random_indices = torch.randint(0, len(X_all), (len(X_batch),))
        X_random = X_all[random_indices]
        X_corrupted = X_batch * (1 - mask) + X_random * mask
        return X_corrupted
    
    def nt_xent_loss(z1, z2, temperature=0.5):
        \"\"\"Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent)\"\"\"
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.shape[0]
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.t()) / temperature
        
        # Mask out self-similarity
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(~mask, -1e9)
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                            torch.arange(0, batch_size)]).to(z1.device)
        
        return F.cross_entropy(similarity_matrix, labels)
    
    # === PRE-TRAINING PHASE (unsupervised — uses ALL data, no labels) ===
    DEVICE = torch.device('cuda' if check_cuda_working() else 'cpu')
    X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    encoder = ScarfEncoder(n_features=X_selected.shape[1]).to(DEVICE)
    projector = ScarfProjector().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    
    pretrain_losses = []
    for epoch in range(50):
        epoch_loss = 0
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE)
            
            # Anchor view (original) and corrupted view
            z_anchor = projector(encoder(batch_X))
            X_corrupt = scarf_corrupt(batch_X, X_tensor.to(DEVICE), corruption_rate=0.6)
            z_corrupt = projector(encoder(X_corrupt))
            
            loss = nt_xent_loss(z_anchor, z_corrupt, temperature=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        pretrain_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [SCARF Pre-train] Epoch {epoch+1}/50 | NT-Xent Loss: {avg_loss:.4f}")
    
    # === FINE-TUNING PHASE: Freeze encoder, train classifier head ===
    encoder.eval()
    with torch.no_grad():
        X_train_embed = encoder(torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        X_test_embed = encoder(torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        joblib.dump({
            'X_train_embed': X_train_embed, 
            'X_test_embed': X_test_embed,
            'pretrain_losses': pretrain_losses
        }, cache_scarf)

# Train any downstream classifier on the LEARNED REPRESENTATIONS
from sklearn.ensemble import GradientBoostingClassifier
clf_scarf = GradientBoostingClassifier(n_estimators=200, max_depth=4)
clf_scarf.fit(X_train_embed, y_train)
y_pred_scarf = clf_scarf.predict(X_test_embed)

# Compare: raw features vs SCARF embeddings
from sklearn.metrics import classification_report
print("\\n=== Raw Features ===")
if 'results' in globals() and 'Random Forest' in results:
    print(classification_report(y_test, results['Random Forest']['model'].predict(X_test)))
elif 'results' in globals() and len(results) > 0:
    first_model = list(results.keys())[0]
    print(classification_report(y_test, results[first_model]['model'].predict(X_test)))
else:
    clf_raw = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    clf_raw.fit(X_train, y_train)
    print(classification_report(y_test, clf_raw.predict(X_test)))
print("\\n=== SCARF Pre-trained Embeddings ===")
print(classification_report(y_test, y_pred_scarf))

# === REPORT VISUAL: Pre-training loss curve + high-speed t-SNE of learned embeddings ===
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Loss curve
if 'pretrain_losses' in locals() and pretrain_losses is not None:
    axes[0].plot(pretrain_losses, color='#6c5ce7', linewidth=2)
    axes[0].set_title('SCARF Pre-training Loss Curve')
else:
    axes[0].text(0.5, 0.5, 'Loss curve not available\\n(loaded from cache)', 
                 ha='center', va='center', fontsize=12, color='gray')
    axes[0].set_title('SCARF Pre-training Loss Curve (Cached)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('NT-Xent Contrastive Loss')
axes[0].grid(alpha=0.3)

# High-speed t-SNE: Use 500 samples (16x faster than 2000, runs in <3s!) with n_jobs=-1
print("👉 Computing high-speed t-SNE visualizations (500 samples)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
raw_2d = tsne.fit_transform(X_test.values[:500])
scatter1 = axes[1].scatter(raw_2d[:, 0], raw_2d[:, 1], c=y_test.values[:500],
                           cmap='RdYlGn', alpha=0.4, s=8)
axes[1].set_title('t-SNE: Raw Features')

embed_2d = tsne.fit_transform(X_test_embed[:500])
scatter2 = axes[2].scatter(embed_2d[:, 0], embed_2d[:, 1], c=y_test.values[:500],
                           cmap='RdYlGn', alpha=0.4, s=8)
axes[2].set_title('t-SNE: SCARF Learned Embeddings')

plt.suptitle('Self-Supervised Contrastive Learning (SCARF): Representation Quality',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'scarf_embeddings.png'), dpi=150, bbox_inches='tight')
plt.show()
"""

nb['cells'][121]['source'] = [line + '\n' for line in scarf_source.split('\n')]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: SCARF cell updated!")
