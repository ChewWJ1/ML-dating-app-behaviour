import json
import os

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

# Load the notebook JSON
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching and injecting high-speed joblib caches programmatically...")

dml_source = """# --- V5.1 CAUSAL TREATMENT EFFECTS: DOUBLE MACHINE LEARNING (DML) ---
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt
import os, joblib

print("🔬 Estimating Causal Treatment Effects via Double Machine Learning...")

os.makedirs('../models_v5', exist_ok=True)
cache_file = '../models_v5/dml_causal.joblib'

if os.path.exists(cache_file):
    print("🚀 Loading pre-computed Causal Double Machine Learning results from cache...")
    cache_data = joblib.load(cache_file)
    ate = cache_data['ate']
    ci_low = cache_data['ci_low']
    ci_high = cache_data['ci_high']
    se = cache_data['se']
    p_val = cache_data['p_val']
else:
    # Treatment Variable: profile_pics_count (High-effort profile presentation)
    # Convert treatment to binary (treatment = user has > median profile pics count)
    T_raw = X['profile_pics_count'].values if hasattr(X, 'columns') else X[:, 6]
    T = (T_raw > np.median(T_raw)).astype(float)
    Y = y.values if hasattr(y, 'values') else y

    # Drop treatment column to get confounders W
    W = X.drop(columns=['profile_pics_count']).values if hasattr(X, 'columns') else np.delete(X, 6, axis=1)

    # Step 1: Propensity score model (Predict Treatment from Confounders)
    print("👉 Step 1: Residualling confounders out of treatment variable...")
    model_T = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    model_T.fit(W, T)
    T_pred = model_T.predict_proba(W)[:, 1]
    T_res = T - T_pred  # Treatment residual

    # Step 2: Outcome model (Predict Outcome from Confounders)
    print("👉 Step 2: Residualling confounders out of matchmaking outcome...")
    model_Y = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    model_Y.fit(W, Y)
    Y_pred = model_Y.predict_proba(W)[:, 1]
    Y_res = Y - Y_pred  # Outcome residual

    # Step 3: Regress residuals to estimate ATE (Average Treatment Effect)
    print("👉 Step 3: Regressing residuals to estimate Average Treatment Effect (ATE)...")
    ate_model = LinearRegression(fit_intercept=False)
    ate_model.fit(T_res.reshape(-1, 1), Y_res)
    ate = ate_model.coef_[0]

    # Step 4: Bootstrap to estimate 95% Confidence Intervals
    print("👉 Running bootstrap iterations for causal significance testing...")
    rng = np.random.default_rng(RANDOM_STATE)
    boot_ates = []
    n_samples = len(T_res)
    for _ in range(100):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_T_res = T_res[idx]
        boot_Y_res = Y_res[idx]
        bm = LinearRegression(fit_intercept=False).fit(boot_T_res.reshape(-1, 1), boot_Y_res)
        boot_ates.append(bm.coef_[0])
        
    se = np.std(boot_ates)
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se
    z_score = ate / se
    p_val = 2 * (1 - norm.cdf(abs(z_score)))
    
    # Save cache
    joblib.dump({
        'ate': ate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'p_val': p_val
    }, cache_file)

print("\\n========================================================")
print("📊 CAUSAL DOUBLE MACHINE LEARNING RESULTS")
print("========================================================")
print(f"👉 Estimated Causal ATE (profile_pics > 3): {ate:.4f}")
print(f"👉 95% Bootstrap Confidence Interval       : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"👉 Standard Error                           : {se:.4f}")
print(f"👉 Causal Effect p-value                     : {p_val:.6f}")
if ci_low > 0:
    print("🌟 Conclusion: Profile photo investment has a SIGNIFICANT POSITIVE CAUSAL EFFECT on matches!")
else:
    print("⚠️ Conclusion: Causal effect is not statistically distinct from zero after controlling for location/income brackets.")
print("========================================================")"""

gnn_source = """# pip install torch-geometric
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import numpy as np
import os, joblib

# Step 3: Graph Attention Network Class Definition (required for cache load)
class DatingGAT(torch.nn.Module):
    def __init__(self, n_features, hidden=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(n_features, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.3)
        self.classifier = torch.nn.Linear(hidden, 2)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x)

os.makedirs('../models_v5', exist_ok=True)
cache_file = '../models_v5/gnn_gat.joblib'

if os.path.exists(cache_file):
    print("🚀 Loading pre-trained GNN (GAT) model and graph from cache...")
    cache_data = joblib.load(cache_file)
    edge_index = cache_data['edge_index']
    test_acc = cache_data['test_acc']
    
    # Rebuild PyG Data object
    data = Data(
        x=torch.tensor(X_selected.values, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y.values, dtype=torch.long)
    )
    # Extract indices from split DataFrames
    train_indices = X_train.index if hasattr(X_train, 'index') else np.arange(len(X_train))
    test_indices = X_test.index if hasattr(X_test, 'index') else np.arange(len(X_train), len(X_train)+len(X_test))

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    data = data.to(DEVICE)
    
    model = DatingGAT(n_features=X_selected.shape[1]).to(DEVICE)
    model.load_state_dict(cache_data['model_state'])
    print(f"GAT Test Accuracy: {test_acc:.4f}")
else:
    print("👉 Fitting Graph Attention Network from scratch...")
    # Step 1: Build KNN graph from feature similarity
    k = 10  # each user connected to 10 most similar users
    adj_matrix = kneighbors_graph(X_selected.values, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)

    # Step 2: Create PyG Data object
    data = Data(
        x=torch.tensor(X_selected.values, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y.values, dtype=torch.long)
    )

    # Extract indices from split DataFrames
    train_indices = X_train.index if hasattr(X_train, 'index') else np.arange(len(X_train))
    test_indices = X_test.index if hasattr(X_test, 'index') else np.arange(len(X_train), len(X_train)+len(X_test))

    # Train/test masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    model = DatingGAT(n_features=X_selected.shape[1]).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Train
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    pred = model(data).argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"GAT Test Accuracy: {test_acc:.4f}")
    
    # Save cache (ensure device is CPU to prevent GPU mapping errors on reload)
    joblib.dump({
        'edge_index': edge_index,
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'test_acc': test_acc
    }, cache_file)"""

dice_source = """# --- V5 METHODOLOGY 6: ALGORITHMIC RECOURSE (DiCE COUNTERFACTUALS) ---
import dice_ml
import pandas as pd
import numpy as np
import os, joblib

print("⚖️ Generating Counterfactual Explanations via DiCE...")

os.makedirs('../models_v5', exist_ok=True)
cache_file = '../models_v5/dice_recourse.joblib'

if os.path.exists(cache_file):
    print("🚀 Loading pre-computed DiCE Counterfactual Explanations from cache...")
    cache_data = joblib.load(cache_file)
    cf_results = cache_data['cf_results']
    target_idx = cache_data['target_idx']
    print(f"👉 Target user index {target_idx} predicted to be 'Ghosted'. Loaded 3 diverse counterfactuals for recourse from cache...")
    cf_results.visualize_as_dataframe(show_only_changes=True)
else:
    # Select best model
    best_recourse_name = None
    for name in ['Random Forest', 'Random Forest (Tuned)', 'XGBoost', 'LightGBM']:
        if name in results:
            best_recourse_name = name
            break

    if best_recourse_name:
        r_entry = results[best_recourse_name]
        model_obj = r_entry.get('model')
        
        if model_obj:
            # Create a combined dataframe for DiCE data mapping
            # Convert X_train back to a dataframe with column names
            X_train_df = pd.DataFrame(X_train, columns=X.columns)
            train_df = X_train_df.copy()
            train_df['target'] = y_train.values if hasattr(y_train, 'values') else y_train
            
            # Map features
            d_dice = dice_ml.Data(dataframe=train_df, 
                                  continuous_features=list(numeric_cols), 
                                  outcome_name='target')
                                  
            # Wrap classifier in DiCE Model
            m_dice = dice_ml.Model(model=model_obj, backend="sklearn")
            
            # Explainer setup using randomized method
            exp_dice = dice_ml.Dice(d_dice, m_dice, method="random")
            
            # Find a test instance predicted to be "Ghosted" (0)
            y_pred = r_entry['pred']
            ghosted_indices = np.where(y_pred == 0)[0]
            
            if len(ghosted_indices) > 0:
                target_idx = ghosted_indices[0]
                
                # Extract target instance
                X_test_df = pd.DataFrame(X_test, columns=X.columns)
                query_instance = X_test_df.iloc[[target_idx]]
                
                print(f"👉 Target user index {target_idx} predicted to be 'Ghosted'. Generating 3 diverse counterfactuals for recourse...")
                
                # Generate recourse paths
                cf_results = exp_dice.generate_counterfactuals(query_instance, total_CFs=3, desired_class=1)
                
                # Visualize the recourse options
                cf_results.visualize_as_dataframe(show_only_changes=True)
                
                # Dump cache
                joblib.dump({'cf_results': cf_results, 'target_idx': target_idx}, cache_file)
            else:
                print("⚠️ No user profiles predicted to be 'Ghosted' found in the test set.")
        else:
            print("⚠️ Fitted model object not found in results.")
    else:
        print("⚠️ No compatible tree-based champion model found to generate recourse.")"""

uplift_source = """# --- V5.1 UPLIFT MODELING: T-LEARNER META-CLASSIFIER ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os, joblib

print("🎯 Fitting Causal Uplift T-Learner Recommender...")

os.makedirs('../models_v5', exist_ok=True)
cache_file = '../models_v5/causal_uplift.joblib'

if os.path.exists(cache_file):
    print("🚀 Loading pre-trained Causal Uplift models and scores from cache...")
    cache_data = joblib.load(cache_file)
    model_treat = cache_data['model_treat']
    model_ctrl = cache_data['model_ctrl']
    uplift_scores = cache_data['uplift_scores']
    prob_ctrl = cache_data['prob_ctrl']
    prob_treat = cache_data['prob_treat']
    print(f"✅ Uplift estimation complete. Mean Uplift: {np.mean(uplift_scores):.4f}")
else:
    # Get treatment variable index dynamically
    pics_col_idx = list(X.columns).index('profile_pics_count')
    T_train_raw = X_train[:, pics_col_idx]
    T_train = (T_train_raw > np.median(T_train_raw)).astype(int)

    # Filter Treatment and Control indices
    idx_treat = np.where(T_train == 1)[0]
    idx_ctrl = np.where(T_train == 0)[0]

    # Fit T-Learner models (Random Forest backbones)
    model_treat = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    model_ctrl = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)

    print("👉 Training treatment-response estimator (M_1)...")
    model_treat.fit(X_train[idx_treat], y_train.values[idx_treat] if hasattr(y_train, 'values') else y_train[idx_treat])

    print("👉 Training control-response estimator (M_0)...")
    model_ctrl.fit(X_train[idx_ctrl], y_train.values[idx_ctrl] if hasattr(y_train, 'values') else y_train[idx_ctrl])

    # Predict Uplift (Individual Treatment Effect) on test set
    prob_treat = model_treat.predict_proba(X_test)[:, 1]
    prob_ctrl = model_ctrl.predict_proba(X_test)[:, 1]
    uplift_scores = prob_treat - prob_ctrl
    
    print(f"✅ Uplift estimation complete. Mean Uplift: {np.mean(uplift_scores):.4f}")
    
    # Save cache
    joblib.dump({
        'model_treat': model_treat,
        'model_ctrl': model_ctrl,
        'uplift_scores': uplift_scores,
        'prob_ctrl': prob_ctrl,
        'prob_treat': prob_treat
    }, cache_file)

# Causal Segmentation mapping
segments = []
for u, c in zip(uplift_scores, prob_ctrl):
    if u > 0.05: segments.append('Persuadable (Target)')
    elif c > 0.60: segments.append('Sure Thing (No action)')
    elif u < -0.05: segments.append('Sleeping Dog (Do not disturb)')
    else: segments.append('Lost Cause (Ignore)')
    
segments = np.array(segments)
seg_counts = pd.Series(segments).value_counts()

# Plot uplift distribution & segmentation
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(uplift_scores, bins=40, kde=True, color='#009688', alpha=0.7)
plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Zero Uplift Boundary')
plt.title('🎯 Causal Uplift Score Distribution (ITE)', fontsize=12, fontweight='bold')
plt.xlabel('Estimated Uplift (Treatment Prob - Control Prob)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.barplot(x=seg_counts.index, y=seg_counts.values, palette='viridis')
plt.title('🔬 App Targeting Segments (Causal Prescriptive)', fontsize=12, fontweight='bold')
plt.ylabel('User Count')
plt.xticks(rotation=15)

plt.suptitle('🎯 Causal Uplift Modeling: Going from Predictive to Prescriptive Recommendations', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""

dml_injected = False
gnn_injected = False
dice_injected = False
uplift_injected = False

for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # Match DML Causal
        if "# --- V5.1 CAUSAL TREATMENT EFFECTS: DOUBLE MACHINE LEARNING (DML) ---" in source:
            print(f"Injecting DML cache in Cell {idx}...")
            cell['source'] = [line + '\n' for line in dml_source.split('\n')]
            dml_injected = True
            
        # Match GNN GAT
        elif "DatingGAT" in source and "torch_geometric" in source:
            print(f"Injecting GNN cache in Cell {idx}...")
            cell['source'] = [line + '\n' for line in gnn_source.split('\n')]
            gnn_injected = True
            
        # Match DiCE
        elif "# --- V5 METHODOLOGY 6: ALGORITHMIC RECOURSE (DiCE COUNTERFACTUALS) ---" in source:
            print(f"Injecting DiCE recourse cache in Cell {idx}...")
            cell['source'] = [line + '\n' for line in dice_source.split('\n')]
            dice_injected = True
            
        # Match Uplift
        elif "# --- V5.1 UPLIFT MODELING: T-LEARNER META-CLASSIFIER ---" in source:
            print(f"Injecting Uplift cache in Cell {idx}...")
            cell['source'] = [line + '\n' for line in uplift_source.split('\n')]
            uplift_injected = True

if all([dml_injected, gnn_injected, dice_injected, uplift_injected]):
    print("All caches successfully mapped!")
    # Write back the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved successfully!")
else:
    print(f"Error mapping caches: DML={dml_injected}, GNN={gnn_injected}, DiCE={dice_injected}, Uplift={uplift_injected}")
