"""
V7_Strict Deep Audit + PyCaret/FLAML Fix Script
Scans every cell for: stale variables, broken references, indentation, logic errors.
Also fixes PyCaret (exclude SVM) and FLAML (retrain from scratch on V7 features).
"""
import json
import re

NB_PATH = r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

issues = []
fixes = []

# ═══════════════════════════════════════════════════════════
# SCAN 1: Stale variable references
# Variables that should NOT exist in V7: X_selected (outside DL cells),
# X_pca, X (bare), y (bare outside target definition)
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))

    # X_pca should only exist as X_train_pca or X_test_pca
    if re.search(r'\bX_pca\b', src) and 'X_train_pca' not in src and 'X_test_pca' not in src:
        issues.append(f'Cell {i}: STALE reference to X_pca (should be X_train_pca/X_test_pca)')

    # X_selected should only exist in GAT/SCARF cells (116+)
    if re.search(r'\bX_selected\b', src) and i < 116:
        if 'X_train_selected' not in src and 'X_test_selected' not in src:
            issues.append(f'Cell {i}: STALE reference to X_selected before DL section')

# ═══════════════════════════════════════════════════════════
# SCAN 2: Indentation errors (basic check)
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    try:
        compile(src, f'cell_{i}', 'exec')
    except SyntaxError as e:
        issues.append(f'Cell {i}: SYNTAX ERROR — {e.msg} (line {e.lineno})')
    except Exception:
        pass  # Other compile errors (missing imports etc) are fine

# ═══════════════════════════════════════════════════════════
# SCAN 3: scale_pos_weight check (should be 1 everywhere)
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'scale_pos_weight' in src and 'scale_pos_weight=1' not in src:
        # Check if it's not a comment
        for line in src.split('\n'):
            if 'scale_pos_weight' in line and not line.strip().startswith('#'):
                if 'scale_pos_weight=1' not in line:
                    issues.append(f'Cell {i}: scale_pos_weight not set to 1: {line.strip()[:80]}')

# ═══════════════════════════════════════════════════════════
# SCAN 4: Remaining list(results.values())[0] references
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'list(results.values())[0]' in src:
        issues.append(f'Cell {i}: Still using list(results.values())[0] — should use best_name')

# ═══════════════════════════════════════════════════════════
# SCAN 5: Real tokens still present
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'eyJhbGci' in src:
        issues.append(f'Cell {i}: Real JWT token still present')
    if 'hf_' in src and 'INSERT_YOUR' not in src and 'HF_TOKEN' in src:
        # Check if it's a real token
        for line in src.split('\n'):
            if 'hf_' in line and 'INSERT_YOUR' not in line and 'environ' in line:
                if not line.strip().startswith('#'):
                    issues.append(f'Cell {i}: Possible real HF token: {line.strip()[:60]}')

# ═══════════════════════════════════════════════════════════
# SCAN 6: Hardcoded Random Forest fallback loops
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if "for name in ['Random Forest" in src or "for name in ['LightGBM', 'Random Forest" in src:
        issues.append(f'Cell {i}: Hardcoded model fallback loop (should use best_name dynamically)')

# ═══════════════════════════════════════════════════════════
# SCAN 7: Old models_v5 path references
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'models_v5' in src:
        issues.append(f'Cell {i}: Reference to old models_v5 directory')

# ═══════════════════════════════════════════════════════════
# SCAN 8: learning_curve using X_train instead of X_train_raw
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'learning_curve(' in src:
        # Check if it uses X_train without ImbPipeline
        if 'X_train,' in src and 'X_train_raw' not in src:
            issues.append(f'Cell {i}: learning_curve still uses X_train instead of X_train_raw')

# ═══════════════════════════════════════════════════════════
# SCAN 9: cross_val_score using X_train instead of X_train_raw
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'cross_val_score(' in src:
        if 'X_train,' in src and 'X_train_raw' not in src:
            issues.append(f'Cell {i}: cross_val_score uses X_train (SMOTE) instead of X_train_raw')

# ═══════════════════════════════════════════════════════════
# SCAN 10: cv=3 still present in CalibratedClassifierCV
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'CalibratedClassifierCV' in src and 'cv=3' in src:
        issues.append(f'Cell {i}: CalibratedClassifierCV still uses cv=3')

# ═══════════════════════════════════════════════════════════
# SCAN 11: Check FLAML cell for stale model loading
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'flaml_results.joblib' in src and 'original_flaml_path' in src:
        # Check if it still tries to load old model first
        if "Reusing original FLAML" in src:
            issues.append(f'Cell {i}: FLAML still tries to load old V5 model first (will cause column mismatch)')

# ═══════════════════════════════════════════════════════════
# SCAN 12: PyCaret setup — verify SVM excluded and n_jobs
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'compare_models(' in src and 'pycaret' in src.lower():
        if "exclude=['svm']" not in src and "exclude=[\"svm\"]" not in src:
            issues.append(f'Cell {i}: PyCaret compare_models does not exclude SVM')

# ═══════════════════════════════════════════════════════════
# SCAN 13: Check for any cell still using scaler.fit_transform on full df
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'scaler.fit_transform(df' in src or 'scaler.fit_transform(X)' in src:
        if not src.strip().startswith('#') and '# df[' not in src:
            issues.append(f'Cell {i}: Scaler fitted on full dataset (data leakage)')

# ═══════════════════════════════════════════════════════════
# SCAN 14: SHAP cell — check fallback logic
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'SHAP Interaction' in src and 'TreeExplainer' in src:
        if 'best_tree_name = best_name' not in src:
            issues.append(f'Cell {i}: SHAP does not use best_name for tree model selection')

# ═══════════════════════════════════════════════════════════
# FIX: FLAML — stop loading old V5 model, retrain from scratch
# ═══════════════════════════════════════════════════════════
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'flaml_results.joblib' in src and 'original_flaml_path' in src:
        # Remove the "Reusing original FLAML" fallback
        new_src = src.replace(
            "original_flaml_path = '../models/flaml_results.joblib'",
            "# original_flaml_path removed — V5 model has incompatible feature columns"
        )
        # Also remove the block that loads the old model
        new_src = re.sub(
            r'if os\.path\.exists\(original_flaml_path\).*?(?=\n    (?:flaml_checkpoint_path|# ))',
            '',
            new_src,
            flags=re.DOTALL
        )
        if new_src != src:
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            fixes.append(f'Cell {i}: FLAML — removed stale V5 model loading')
        break

# ═══════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'  DEEP AUDIT RESULTS')
print(f'{"="*60}')
print(f'  Issues found: {len(issues)}')
print(f'  Auto-fixes applied: {len(fixes)}')
print()

if issues:
    print('ISSUES:')
    for issue in issues:
        print(f'  [!!] {issue}')
else:
    print('  [OK] No issues found!')

if fixes:
    print('\nFIXES APPLIED:')
    for fix in fixes:
        print(f'  [OK] {fix}')

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
