import json

notebook_path = 'notebooks/ML_dating_app_behaviour V8.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper function to find a cell containing a specific string
def find_cell(query):
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if query in source:
                return cell
    return None

# 1. Threshold Optimization
cell_calib = find_cell("calibrated_clf = CalibratedClassifierCV")
if cell_calib:
    source = "".join(cell_calib['source'])
    old_pr_code = """    target_dict[champion_name]['y_prob'] = calibrated_clf.predict_proba(X_eval)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_eval, target_dict[champion_name]['y_prob'])"""
    new_pr_code = """    # FIX: Calculate threshold on calibration set to avoid leakage
    probs_calib = calibrated_clf.predict_proba(X_calib)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_calib, probs_calib)
    
    target_dict[champion_name]['y_prob'] = calibrated_clf.predict_proba(X_eval)[:, 1]"""
    source = source.replace(old_pr_code, new_pr_code)
    cell_calib['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 2. FGSM
cell_fgsm = find_cell("def fgsm_attack")
if cell_fgsm:
    source = "".join(cell_fgsm['source'])
    old_fgsm_step = """    # Perturb in the direction of the gradient sign
    perturbation = epsilon * X_adv.grad.sign()"""
    new_fgsm_step = """    # Perturb in the direction of the gradient sign
    # FIX: Only perturb continuous numerical features (mask categorical)
    mask = torch.zeros(X.shape[1], dtype=torch.float32).to(device)
    if 'numeric_cols' in globals() and 'feature_names' in globals():
        for i, c in enumerate(feature_names):
            if c in numeric_cols: mask[i] = 1.0
    else:
        mask[:19] = 1.0 # fallback
    perturbation = epsilon * X_adv.grad.sign() * mask"""
    source = source.replace(old_fgsm_step, new_fgsm_step)
    cell_fgsm['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 3. DiCE
cell_dice = find_cell("exp_dice.generate_counterfactuals")
if cell_dice:
    source = "".join(cell_dice['source'])
    old_dice = """                # Generate recourse paths
                cf_results = exp_dice.generate_counterfactuals(query_instance, total_CFs=3, desired_class=1)"""
    new_dice = """                # Generate recourse paths
                # FIX: Restrict to mutable features
                mutable_features = [c for c in ['profile_pics_count', 'bio_length', 'message_sent_count', 'app_usage_time_min', 'swipe_right_ratio', 'emoji_usage_rate'] if c in train_df.columns]
                cf_results = exp_dice.generate_counterfactuals(query_instance, total_CFs=3, desired_class=1, features_to_vary=mutable_features)"""
    source = source.replace(old_dice, new_dice)
    cell_dice['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 4. Knowledge Distillation
cell_kd = find_cell("distill_losses = []")
if cell_kd:
    source = "".join(cell_kd['source'])
    old_kd = """distill_losses = []
for epoch in range(100):
    student.train()
    logits = student(X_train_t)
    
    # Distillation loss: KL divergence between soft teacher and student outputs
    student_soft = torch.sigmoid(logits / temperature)
    teacher_soft_scaled = y_soft  # teacher probs already soft
    distill_loss = F.binary_cross_entropy(student_soft, teacher_soft_scaled)
    
    # Hard label loss: standard BCE
    hard_loss = F.binary_cross_entropy_with_logits(logits, y_hard)
    
    # Combined loss
    loss = alpha * (temperature ** 2) * distill_loss + (1 - alpha) * hard_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    distill_losses.append(loss.item())"""
    new_kd = """# FIX: Mini-batching for Student Network
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(X_train_t, y_hard, y_soft)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
distill_losses = []
for epoch in range(20):
    student.train()
    for b_x, b_y_hard, b_y_soft in dataloader:
        logits = student(b_x)
        student_soft = torch.sigmoid(logits / temperature)
        distill_loss = F.binary_cross_entropy(student_soft, b_y_soft)
        hard_loss = F.binary_cross_entropy_with_logits(logits, b_y_hard)
        loss = alpha * (temperature ** 2) * distill_loss + (1 - alpha) * hard_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    distill_losses.append(loss.item())"""
    source = source.replace(old_kd, new_kd)
    cell_kd['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 5. T-Learner IPW
cell_tlearner = find_cell("model_treat.fit(")
if cell_tlearner:
    source = "".join(cell_tlearner['source'])
    old_tlearner = """    print("👉 Training treatment-response estimator (M_1)...")
    model_treat.fit((X_train.iloc[idx_treat] if isinstance(X_train, pd.DataFrame) else X_train[idx_treat]), y_train.values[idx_treat] if hasattr(y_train, 'values') else y_train[idx_treat])

    print("👉 Training control-response estimator (M_0)...")
    model_ctrl.fit((X_train.iloc[idx_ctrl] if isinstance(X_train, pd.DataFrame) else X_train[idx_ctrl]), y_train.values[idx_ctrl] if hasattr(y_train, 'values') else y_train[idx_ctrl])"""
    new_tlearner = """    # FIX: Inverse Probability Weighting (IPW) for observational causal inference
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    ps_model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    if isinstance(X_train, pd.DataFrame):
        W_train = X_train.drop(columns=['profile_pics_count'])
    else:
        W_train = np.delete(X_train, pics_col_idx, axis=1)
    ps_scores = cross_val_predict(ps_model, W_train, T_train, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
    ps_scores = np.clip(ps_scores, 0.05, 0.95)
    weights = np.zeros_like(T_train, dtype=float)
    weights[idx_treat] = 1.0 / ps_scores[idx_treat]
    weights[idx_ctrl] = 1.0 / (1.0 - ps_scores[idx_ctrl])

    print("👉 Training treatment-response estimator (M_1) with IPW...")
    model_treat.fit((X_train.iloc[idx_treat] if isinstance(X_train, pd.DataFrame) else X_train[idx_treat]), y_train.values[idx_treat] if hasattr(y_train, 'values') else y_train[idx_treat], sample_weight=weights[idx_treat])

    print("👉 Training control-response estimator (M_0) with IPW...")
    model_ctrl.fit((X_train.iloc[idx_ctrl] if isinstance(X_train, pd.DataFrame) else X_train[idx_ctrl]), y_train.values[idx_ctrl] if hasattr(y_train, 'values') else y_train[idx_ctrl], sample_weight=weights[idx_ctrl])"""
    source = source.replace(old_tlearner, new_tlearner)
    cell_tlearner['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# 6. PC Algorithm
cell_pc = find_cell("indep_test='fisherz'")
if cell_pc:
    source = "".join(cell_pc['source'])
    old_pc = """# Run PC algorithm for causal discovery
cg = pc(data_causal, alpha=0.05, indep_test='fisherz')"""
    new_pc = """# Run PC algorithm for causal discovery
# FIX: Downsample and use kci test (non-parametric)
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(data_causal), size=min(1500, len(data_causal)), replace=False)
data_causal_sampled = data_causal[sample_idx]
cg = pc(data_causal_sampled, alpha=0.05, indep_test='kci')"""
    source = source.replace(old_pc, new_pc)
    cell_pc['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

with open('notebooks/ML_dating_app_behaviour V8_patched.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
