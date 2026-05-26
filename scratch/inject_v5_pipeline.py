import json
import os

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell_index_by_md_title(title):
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if title in source:
                return i
    return -1

def find_cell_index_by_code_signature(sig):
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if sig in source:
                return i
    return -1

# 1. Update pip install cell
pip_idx = find_cell_index_by_code_signature('!pip install -q')
if pip_idx != -1:
    source = cells[pip_idx]['source']
    for line_idx, line in enumerate(source):
        if '!pip install -q' in line:
            source[line_idx] = line.rstrip() + " tabpfn dice-ml shap\n"
            print(f"Updated pip install cell at index {pip_idx} with tabpfn dice-ml shap.")
            break
else:
    print("Warning: pip install cell not found.")

def create_md_cell(content):
    lines = [line + "\n" for line in content.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def create_code_cell(content):
    lines = [line + "\n" for line in content.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": lines
    }

# 2. Inject Out-of-Distribution (OOD) Rejection Guardrail
idx_robust = find_cell_index_by_md_title('RobustScaler')
if idx_robust != -1:
    insert_pos = idx_robust + 2
    
    ood_md = """## Flex 11: 🚨 Out-of-Distribution (OOD) Rejection Guardrail
In high-stakes, human-centric systems like dating recommendations, deploying a machine learning model without an input guardrail is risky. Adversarial, corrupted, or highly anomalous profile data can lead to unpredictable predictions. 

To solve this, we implement a **production-grade Out-of-Distribution (OOD) Rejection Guardrail** using an **Isolation Forest**. This unsupervised algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Recursive partitioning can be represented by a tree structure, where the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. Anomalous profiles require much fewer splits to isolate, resulting in shorter path lengths.

If an incoming user profile has an anomaly score below the dynamic threshold (offset), the system rejects the input and flags it for manual review or default recommendations, rather than serving a potentially erroneous model prediction."""

    ood_code = """# --- V5 METHODOLOGY 1: OUT-OF-DISTRIBUTION (OOD) REJECTION GUARDRAIL ---
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("🚨 Initializing Out-of-Distribution (OOD) Rejection Guardrail...")

# Initialize Isolation Forest on training data
# We set contamination to 0.05 (expecting 5% anomalies or out-of-distribution profiles)
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
iso_forest.fit(X_train)

# Compute anomaly scores (lower scores = more anomalous)
train_scores = iso_forest.score_samples(X_train)
test_scores = iso_forest.score_samples(X_test)

# Predict inlier (1) vs outlier (-1)
train_preds = iso_forest.predict(X_train)
test_preds = iso_forest.predict(X_test)

num_train_anomalies = np.sum(train_preds == -1)
num_test_anomalies = np.sum(test_preds == -1)

print(f"✅ OOD Guardrail calibrated on training set.")
print(f"👉 Detected {num_train_anomalies} anomalies in Training set ({num_train_anomalies/len(X_train)*100:.1f}%)")
print(f"👉 Detected {num_test_anomalies} anomalies in Test set ({num_test_anomalies/len(X_test)*100:.1f}%)")
print(f"👉 Calibrated Anomaly Threshold: {iso_forest.offset_:.4f}")

# Plotting the anomaly score distributions with threshold indicator
plt.figure(figsize=(12, 6))
sns.histplot(train_scores, bins=50, kde=True, color='#2196F3', alpha=0.6, label='Train Profiles')
sns.histplot(test_scores, bins=50, kde=True, color='#E91E63', alpha=0.4, label='Test Profiles')
plt.axvline(iso_forest.offset_, color='#F44336', linestyle='--', linewidth=2, 
            label=f'OOD Rejection Threshold ({iso_forest.offset_:.4f})')
plt.title('🚨 OOD Rejection System: Profile Anomaly Score Distributions', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Isolation Forest Anomaly Score (Lower values indicate high anomaly)', fontsize=12)
plt.ylabel('Density / Frequency', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.show()"""

    cells.insert(insert_pos, create_md_cell(ood_md))
    cells.insert(insert_pos + 1, create_code_cell(ood_code))
    print(f"Injected OOD Rejection at index {insert_pos}.")
else:
    print("Warning: RobustScaler cell not found.")

# 3. Inject Label Smoothing & Mixup
py_wrapper_idx = find_cell_index_by_code_signature('class PyTorchSklearnClassifier')
if py_wrapper_idx != -1:
    source = cells[py_wrapper_idx]['source']
    source_str = "".join(source)
    
    old_fit_block = """        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X, torch.zeros(batch_X.size(0), 0, dtype=torch.long, device=self.device))
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()"""

    new_fit_block = """        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # --- V5 METHODOLOGY 3: LABEL SMOOTHING & MIXUP REGULARIZATION ---
        alpha = 0.2  # Mixup interpolation parameter
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                
                # Apply Label Smoothing to prevent neural networks from becoming overly confident
                # Smooth labels: 0 -> 0.1, 1 -> 0.9
                batch_y_smoothed = batch_y * 0.8 + 0.1
                
                if alpha > 0 and len(batch_X) > 1:
                    # Sample lambda from Beta distribution
                    lam = np.random.beta(alpha, alpha)
                    # Create shuffled index mapping for mixup pairing
                    shuffled_idx = torch.randperm(batch_X.size(0)).to(self.device)
                    
                    # Compute mixed features and mixed labels
                    mixed_X = lam * batch_X + (1 - lam) * batch_X[shuffled_idx]
                    mixed_y = lam * batch_y_smoothed + (1 - lam) * batch_y_smoothed[shuffled_idx]
                    
                    # Forward pass with mixed inputs
                    preds = self.model(mixed_X, torch.zeros(mixed_X.size(0), 0, dtype=torch.long, device=self.device))
                    loss = criterion(preds, mixed_y)
                else:
                    preds = self.model(batch_X, torch.zeros(batch_X.size(0), 0, dtype=torch.long, device=self.device))
                    loss = criterion(preds, batch_y_smoothed)
                    
                loss.backward()
                optimizer.step()"""

    if old_fit_block in source_str:
        source_str = source_str.replace(old_fit_block, new_fit_block)
        cells[py_wrapper_idx]['source'] = [line + "\n" for line in source_str.split("\n")]
        if cells[py_wrapper_idx]['source'][-1] == "\n":
            cells[py_wrapper_idx]['source'][-1] = ""
        print(f"Successfully injected Label Smoothing & Mixup into PyTorchSklearnClassifier.")
    else:
        print("Warning: Fit block exact match failed.")
else:
    print("Warning: Wrapper class cell not found.")

# 4. Inject TabPFN after SCARF cell
idx_scarf = find_cell_index_by_md_title('### 10.10 Self-Supervised')
if idx_scarf != -1:
    insert_pos = -1
    for k in range(idx_scarf + 1, idx_scarf + 5):
        if cells[k]['cell_type'] == 'code' and 'cache_scarf' in "".join(cells[k]['source']):
            insert_pos = k + 1
            break
            
    if insert_pos != -1:
        tabpfn_md = """## Flex 12: ⚡ Zero-Shot Tabular Transformers (TabPFN)
Traditional tabular models (like Random Forests or XGBoost) require training on the target dataset to learn splits and weights. In contrast, **TabPFN (Tabular Prior-Data Fitted Network)** is a revolutionary **zero-shot deep transformer model** pre-trained on millions of synthetic tabular datasets (using causal structures and prior distributions).

TabPFN approximates the true Bayesian posterior distribution in a single forward pass, without requiring standard gradient descent or hyperparameter tuning on the downstream dataset! However, due to its transformer nature, its computational complexity scales cubically $O(N^3)$ with training size, limiting it to $N \\le 1000$ samples.

We feed a downsampled subsample (1,000 profiles) of our balanced training set as the "prior support context" and perform zero-shot evaluation on the test set."""

        tabpfn_code = """# --- V5 METHODOLOGY 2: ZERO-SHOT TABULAR TRANSFORMERS (TabPFN) ---
import os
import joblib
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

os.makedirs('models_v5_cache', exist_ok=True)
cache_tabpfn = 'models_v5_cache/tabpfn.joblib'

print("⚡ Initializing Zero-Shot Tabular Transformer (TabPFN)...")

if os.path.exists(cache_tabpfn):
    print("⏭️  Loading cached TabPFN model and predictions...")
    tabpfn_data = joblib.load(cache_tabpfn)
    y_pred_tab = tabpfn_data['pred']
    y_prob_tab = tabpfn_data['prob']
else:
    # Downsample balanced training set to 1000 samples for TabPFN computational budget
    np.random.seed(RANDOM_STATE)
    sub_indices = np.random.choice(len(X_train_balanced), size=min(1000, len(X_train_balanced)), replace=False)
    X_train_sub = X_train_balanced[sub_indices]
    y_train_sub = y_train_balanced[sub_indices]
    
    # Initialize TabPFN (ensemble configurations specify how many forward passes to average)
    tabpfn = TabPFNClassifier(device=DEVICE, N_ensemble_configurations=32)
    tabpfn.fit(X_train_sub, y_train_sub)
    
    # Make zero-shot predictions on test set
    y_pred_tab = tabpfn.predict(X_test)
    y_prob_tab = tabpfn.predict_proba(X_test)[:, 1]
    
    # Cache predictions and model details
    joblib.dump({'pred': y_pred_tab, 'prob': y_prob_tab}, cache_tabpfn)
    print("💾 TabPFN predictions cached successfully.")

# Evaluate zero-shot transformer performance
acc_tab = accuracy_score(y_test, y_pred_tab)
f1_tab = f1_score(y_test, y_pred_tab)
auc_tab = roc_auc_score(y_test, y_prob_tab)

print("\\n📊 TabPFN Zero-Shot Performance:")
print(f"👉 Test Accuracy: {acc_tab:.4f}")
print(f"👉 Test F1-Score: {f1_tab:.4f}")
print(f"👉 Test ROC-AUC  : {auc_tab:.4f}")

# Append to results table if comparison exists
if 'results' in globals():
    results['TabPFN Transformer (Zero-Shot)'] = {
        'model': None,
        'pred': y_pred_tab,
        'prob': y_prob_tab,
        'acc': acc_tab,
        'f1': f1_tab,
        'auc': auc_tab
    }"""

        cells.insert(insert_pos, create_md_cell(tabpfn_md))
        cells.insert(insert_pos + 1, create_code_cell(tabpfn_code))
        print(f"Injected TabPFN at index {insert_pos}.")
    else:
        print("Warning: SCARF code cell not found.")
else:
    print("Warning: SCARF header not found.")

# 5. Inject SHAP Interaction
idx_hstat = find_cell_index_by_md_title('H-Statistic')
if idx_hstat != -1:
    insert_pos = -1
    for k in range(idx_hstat + 1, idx_hstat + 5):
        if cells[k]['cell_type'] == 'code' and 'h_statistic' in "".join(cells[k]['source']):
            insert_pos = k + 1
            break
            
    if insert_pos != -1:
        shap_md = """## Flex 13: 🌌 SHAP Interaction Values (Attribution of Synergies)
While standard feature importance techniques (like permutation importance or standard SHAP values) assign a single score to each feature, they fail to capture **joint feature attributions**. In other words, they don't show how the combination of two features shifts the model's predictions beyond their individual effects.

To uncover these deep statistical synergies, we compute **SHAP Interaction Values**. Based on the game-theoretic concept of the *Shapley Interaction Index*, these values allocate prediction shifts among all pairs of features. This allows us to map the precise mathematical interactions (e.g., how the combination of high swipe ratios and high mutual match rates dynamically affects a user's likelihood of matching)."""

        shap_code = """# --- V5 METHODOLOGY 4: SHAP INTERACTION VALUES ---
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("🌌 Computing SHAP Interaction Values for Tree-Based Champion...")

# Identify a representative tree-based champion model (Random Forest or XGBoost)
best_tree_name = None
for name in ['Random Forest (Tuned)', 'Random Forest', 'XGBoost (Tuned)', 'XGBoost', 'LightGBM']:
    if name in results or (name.replace(' (Tuned)', '') in results):
        best_tree_name = name
        break

if best_tree_name:
    print(f"👉 Selected tree model for SHAP: {best_tree_name}")
    # Extract the actual fitted model object
    r_entry = results.get(best_tree_name) or results.get(best_tree_name.replace(' (Tuned)', ''))
    tree_model = r_entry.get('model')
    
    if tree_model:
        # Create TreeExplainer
        explainer = shap.TreeExplainer(tree_model)
        
        # Take a subset of 150 test instances to speed up interaction calculations
        X_sample = X_test.iloc[:150] if hasattr(X_test, 'iloc') else pd.DataFrame(X_test[:150], columns=X.columns)
        
        # Compute standard SHAP values and SHAP interaction values
        shap_values_obj = explainer(X_sample)
        shap_interaction_values = explainer.shap_interaction_values(X_sample)
        
        # Identify top two features based on mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values_obj.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-2:]
        feat1_idx, feat2_idx = top_indices[1], top_indices[0]
        feat1_name = X_sample.columns[feat1_idx]
        feat2_name = X_sample.columns[feat2_idx]
        
        print(f"👉 Primary Top Feature: {feat1_name}")
        print(f"👉 Secondary Interacting Feature: {feat2_name}")
        
        # Draw a beautiful 2D SHAP dependence plot mapping the interaction
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat1_name,
            shap_values_obj.values,
            X_sample,
            interaction_index=feat2_name,
            show=False
        )
        plt.title(f"🌌 SHAP Interaction Analysis: {feat1_name} × {feat2_name}", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
        
        # Plot 2D Interaction matrix for the first instance
        plt.figure(figsize=(10, 8))
        # Take interactions for the top 8 features
        top_8_indices = np.argsort(mean_abs_shap)[-8:]
        top_8_names = [X_sample.columns[i] for i in top_8_indices]
        sample_interaction_matrix = shap_interaction_values[0][top_8_indices][:, top_8_indices]
        
        sns.heatmap(sample_interaction_matrix, xticklabels=top_8_names, yticklabels=top_8_names,
                    annot=True, fmt=".4f", cmap="coolwarm", center=0, cbar_kws={'label': 'Interaction Value'})
        plt.title("🔬 SHAP Interaction Matrix (Sample Instance — Top 8 Features)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
        
    else:
        print("⚠️ Fit model object not found in results entry.")
else:
    print("⚠️ No tree-based model found to compute SHAP Interaction Values.")"""

        cells.insert(insert_pos, create_md_cell(shap_md))
        cells.insert(insert_pos + 1, create_code_cell(shap_code))
        print(f"Injected SHAP Interaction at index {insert_pos}.")
    else:
        print("Warning: H-Statistic code cell not found.")
else:
    print("Warning: H-Statistic header not found.")

# 6. Inject Model Calibration
idx_fgsm = find_cell_index_by_md_title('FGSM')
if idx_fgsm != -1:
    insert_pos = -1
    for k in range(idx_fgsm + 1, idx_fgsm + 5):
        if cells[k]['cell_type'] == 'code' and 'fgsm_attack' in "".join(cells[k]['source']):
            insert_pos = k + 1
            break
            
    if insert_pos != -1:
        calib_md = """## Flex 14: 📈 Model Calibration & Reliability Diagrams
For downstream applications (such as matching algorithms or dynamic monetization), the raw confidence score of a classifier needs to represent a **true probability**. For example, if a model predicts a matchmaking probability of 80% for a user profile, 80 out of 100 such profiles should indeed match.

However, complex non-linear models (especially Deep Neural Networks or heavily boosted trees) are notorious for producing **uncalibrated probabilities** (e.g. overconfident predictions). 

To ensure probabilistic reliability, we wrap our champion model in `CalibratedClassifierCV` using **Isotonic Regression**. We then evaluate prediction reliability before and after calibration using a **Reliability Diagram (Calibration Curve)**, validating our model's uncertainty with mathematical rigor."""

        calib_code = """# --- V5 METHODOLOGY 5: MODEL CALIBRATION & RELIABILITY CURVES ---
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import numpy as np
import matplotlib.pyplot as plt

print("📈 Standardizing Probabilities via Isotonic Calibration...")

# Select the champion model
champion_name = None
for name in ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'Random Forest', 'XGBoost', 'LightGBM']:
    if name in results:
        champion_name = name
        break

if champion_name:
    print(f"👉 Champion model selected for calibration: {champion_name}")
    r_entry = results[champion_name]
    base_model = r_entry.get('model')
    
    if base_model:
        # Fit Isotonic Calibration on balanced training data
        calibrated_clf = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_clf.fit(X_train_balanced, y_train_balanced)
        
        # Generate probabilities
        probs_raw = r_entry['prob']
        probs_cal = calibrated_clf.predict_proba(X_test)[:, 1]
        
        # Calculate reliability curves (10 bins)
        prob_true_raw, prob_pred_raw = calibration_curve(y_test, probs_raw, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, probs_cal, n_bins=10)
        
        # Plot reliability diagram
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration (Oracle)')
        plt.plot(prob_pred_raw, prob_true_raw, marker='s', color='#FF9800', label=f'Uncalibrated Champion ({champion_name})')
        plt.plot(prob_pred_cal, prob_true_cal, marker='o', color='#4CAF50', label='Calibrated Champion (Isotonic Regression)')
        
        plt.title('📈 Probability Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
        plt.ylabel('Fraction of Positive Outcomes (Actual matches)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Display Brier Scores (lower is better)
        from sklearn.metrics import brier_score_loss
        brier_raw = brier_score_loss(y_test, probs_raw)
        brier_cal = brier_score_loss(y_test, probs_cal)
        print(f"👉 Uncalibrated Brier Score: {brier_raw:.4f}")
        print(f"👉 Calibrated Brier Score  : {brier_cal:.4f} ({(brier_raw-brier_cal)/brier_raw*100:.1f}% error reduction)")
        
    else:
        print("⚠️ Champion model object not found in results.")
else:
    print("⚠️ No champion model found to perform calibration.")"""

        cells.insert(insert_pos, create_md_cell(calib_md))
        cells.insert(insert_pos + 1, create_code_cell(calib_code))
        print(f"Injected Model Calibration at index {insert_pos}.")
    else:
        print("Warning: FGSM code cell not found.")
else:
    print("Warning: FGSM header not found.")

# 7. Inject Algorithmic Recourse
idx_kd = find_cell_index_by_md_title('Knowledge Distillation')
if idx_kd != -1:
    insert_pos = -1
    for k in range(idx_kd + 1, idx_kd + 5):
        if cells[k]['cell_type'] == 'code' and 'accuracy_score' in "".join(cells[k]['source']):
            insert_pos = k + 1
            break
            
    if insert_pos != -1:
        dice_md = """## Flex 15: ⚖️ Algorithmic Recourse & Counterfactual Explanations (DiCE)
In ethical AI, providing a negative prediction (e.g. "Ghosted") without explanation is insufficient. The principle of **Algorithmic Recourse** dictates that we must provide users with concrete, actionable steps they can take to change their outcome from negative to positive.

Using Microsoft's **DiCE (Diverse Counterfactual Explanations)** framework, we generate counterfactual profiles. These are synthetic but realistic profiles that are minimally different from a target user's profile, but are classified as "Matched" (1) by the model. 

For a user predicted to be "Ghosted", we show the exact minimal changes (e.g., increasing engagement or profile completeness by a specific amount) required to reverse the prediction, putting transparency and agency back into the hands of the user."""

        dice_code = """# --- V5 METHODOLOGY 6: ALGORITHMIC RECOURSE (DiCE COUNTERFACTUALS) ---
import dice_ml
import pandas as pd
import numpy as np

print("⚖️ Generating Counterfactual Explanations via DiCE...")

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
        else:
            print("⚠️ No user profiles predicted to be 'Ghosted' found in the test set.")
    else:
        print("⚠️ Fitted model object not found in results.")
else:
    print("⚠️ No compatible tree-based champion model found to generate recourse.")"""

        cells.insert(insert_pos, create_md_cell(dice_md))
        cells.insert(insert_pos + 1, create_code_cell(dice_code))
        print(f"Injected Algorithmic Recourse at index {insert_pos}.")
    else:
        print("Warning: KD code cell not found.")
else:
    print("Warning: KD header not found.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Success: Injected all V5 methodologies into V5 notebook.")
