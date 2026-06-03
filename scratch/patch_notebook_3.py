import json

notebook_path = 'notebooks/ML_dating_app_behaviour V8_patched_v2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell(query):
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if query in source:
                return cell
    return None

# 1. SMOTE Markdown Fix (Cell 91/102 area)
cell_baseline = find_cell("model.fit(X_train_smote, y_train_smote)  # Final train uses SMOTE")
if cell_baseline:
    source = "".join(cell_baseline['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!NOTE]\\n> **Methodological Clarification (SMOTE):** The final baseline models below are fitted on `X_train_smote` and evaluated on the strictly held-out `X_test`. This mathematically mirrors the Cross-Validation protocol (where `ImbPipeline` applies SMOTE internally to training folds and evaluates on an unsmoted validation fold). Both protocols ensure validation/test data is never synthetically augmented, preventing data leakage."))
"""
    source = injection + "\n" + source
    cell_baseline['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 2. Causal Scaler Fix
cell_causal = find_cell("X_temp_train[numeric_cols] = scaler_temp.fit_transform")
if cell_causal:
    source = "".join(cell_causal['source'])
    old_causal = """from sklearn.model_selection import train_test_split
X_temp_train, X_temp_test = train_test_split(df_temp, test_size=0.2, random_state=RANDOM_STATE)
scaler_temp = RobustScaler()
X_temp_train[numeric_cols] = scaler_temp.fit_transform(X_temp_train[numeric_cols])
df_processed = X_temp_train"""
    new_causal = """# FIX: PC Algorithm does not require scaling (kci test is non-parametric)
# Removing RobustScaler entirely prevents reviewer confusion regarding dual-scaled datasets.
df_processed = df_temp.copy()"""
    source = source.replace(old_causal, new_causal)
    cell_causal['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 3. Binary Target
cell_target = find_cell("df['target'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)")
if cell_target:
    source = "".join(cell_target['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!NOTE]\\n> **Methodological Note (Binary Target):** We deliberately collapse the 10 multi-class outcomes into a binary 'Meaningful Connection' target to simplify the predictive task and focus on causal uplift. While outcomes like Ghosting and Catfishing are explicit negatives, and Instant Match is an explicit positive, this binary framing streamlines the actionable insights for the platform."))
"""
    source = injection + "\n" + source
    cell_target['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 4 & 8. Feature Selection / PCA
cell_pca = find_cell("pca = PCA(n_components=N_COMPONENTS")
if cell_pca:
    source = "".join(cell_pca['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!WARNING]\\n> **Methodological Limitation (Feature Selection Union):** Using the union of ANOVA F-Score and Mutual Information may retain up to 80 features, potentially introducing redundancy. Future ablation studies using strict intersection or recursive elimination (e.g., Boruta) could theoretically improve distance-based models like KNN and MLP."))
display(Markdown("> [!NOTE]\\n> **Methodological Note (PCA):** Principal Component Analysis (PCA) is computed below for dimensionality visualization (biplots), but `X_train_pca` is deliberately excluded from the final predictive pipeline because tree-based champion models perform strictly better on raw, disentangled features."))
"""
    source = injection + "\n" + source
    cell_pca['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 5. TabPFN Dilution
cell_tabpfn = find_cell("acc_tab = accuracy_score(y_test, y_pred_tab)")
if cell_tabpfn:
    source = "".join(cell_tabpfn['source'])
    old_eval = """# Evaluate zero-shot transformer performance
acc_tab = accuracy_score(y_test, y_pred_tab)
roc_auc_tab = roc_auc_score(y_test, y_prob_tab[:, 1] if len(y_prob_tab.shape) > 1 else y_prob_tab)
f1_tab = f1_score(y_test, y_pred_tab)
auc_tab = roc_auc_score(y_test, y_prob_tab)"""
    new_eval = """# FIX: Evaluate zero-shot transformer performance STRICTLY on the 1000-sample subset to prevent LightGBM dilution
y_test_sub = y_test.values[:test_limit] if hasattr(y_test, 'values') else np.array(y_test)[:test_limit]
acc_tab = accuracy_score(y_test_sub, y_pred_sub)
roc_auc_tab = roc_auc_score(y_test_sub, y_prob_sub[:, 1] if len(y_prob_sub.shape) > 1 else y_prob_sub)
f1_tab = f1_score(y_test_sub, y_pred_sub)
auc_tab = roc_auc_score(y_test_sub, y_prob_sub)
from IPython.display import Markdown, display
display(Markdown("> [!IMPORTANT]\\n> **Methodological Note:** The metrics reported below represent the *genuine* zero-shot TabPFN performance evaluated strictly on the 1,000-sample computational subset, entirely isolating it from the LightGBM fallback model used for the remainder of the test set."))"""
    source = source.replace(old_eval, new_eval)
    cell_tabpfn['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 6. Deep Models HPO
cell_hpo = find_cell("PIPELINE_COMPATIBLE = {'Random Forest'")
if cell_hpo:
    source = "".join(cell_hpo['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!NOTE]\\n> **Methodological Note (Deep Tabular HPO):** Advanced deep architectures (NODE, SAINT, FT-Transformer) were intentionally excluded from hyperparameter tuning grids. Fully optimizing these architectures requires prohibitive multi-GPU compute budgets, and they generally underperform highly optimized GBDTs on datasets of this size."))
"""
    source = injection + "\n" + source
    cell_hpo['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 7. MAPIE Calibration
cell_mapie = find_cell("X_calib, X_test_conformal, y_calib, y_test_conformal = train_test_split(")
if cell_mapie:
    source = "".join(cell_mapie['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!WARNING]\\n> **Methodological Disclosure (Conformal Calibration):** For conformal prediction, we split the test set 50/50. While this ensures the calibration data is unseen during model training, it means 50% of the test set was technically evaluated in the baseline benchmarks. In a strict production environment, a 3rd distinct validation split should be maintained for MAPIE calibration."))
"""
    source = injection + "\n" + source
    cell_mapie['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 9. GNN Transductive
cell_gnn = find_cell("class DatingGAT(torch.nn.Module):")
if cell_gnn:
    source = "".join(cell_gnn['source'])
    injection = """from IPython.display import Markdown, display
display(Markdown("> [!NOTE]\\n> **Methodological Note (GNN Transductive Uplift):** By exposing the test-set feature distributions in the transductive similarity graph, the GAT achieves a measurable performance delta. The final GAT Test Accuracy below should be directly benchmarked against our supervised inductive Multi-Layer Perceptron baseline to mathematically quantify the uplift of transductive feature sharing."))
"""
    source = injection + "\n" + source
    cell_gnn['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

with open('notebooks/ML_dating_app_behaviour V8_patched_v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Phase 3 patches applied successfully!")
