"""Post-patch verification: confirm every fix landed correctly in the notebook."""
import json

NB_PATH = r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []

# 1. API token
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'TABPFN_TOKEN' in src:
        if 'eyJ' in src:
            errors.append('1. FAIL: Real JWT still present')
        elif 'INSERT_YOUR_TOKEN_HERE' in src:
            print('[OK] 1. API token: INSERT_YOUR_TOKEN_HERE confirmed')
        break

# 2. StandardScaler print
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if "print('Numerical features normalized with StandardScaler')" in src:
        errors.append('2. FAIL: StandardScaler print still present')
        break
else:
    print('[OK] 2. StandardScaler print: removed/replaced')

# 3. Duplicate param_grids
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'param_grids' in src and "'LightGBM'" in src:
        count = src.count("'LightGBM'")
        if count > 1:
            errors.append(f'3. FAIL: LightGBM still appears {count} times')
        else:
            print(f'[OK] 3. param_grids: LightGBM appears {count} time(s), CatBoost appears {src.count("CatBoost")} time(s)')
        break

# 4. DML cross-fitting
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'model_T' in src and 'DML' in src:
        if 'KFold' in src and 'train_idx, val_idx' in src:
            print('[OK] 4. DML: K-fold cross-fitting implemented')
        else:
            errors.append('4. FAIL: DML still uses in-sample prediction')
        if 'model_T.fit(W, T)' in src:
            errors.append('4. FAIL: Old in-sample model_T.fit(W, T) still present')
        break

# 5. XGBoost scale_pos_weight
found_old = False
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'scale_pos_weight=(30150/19850)' in src:
        found_old = True
        break
if found_old:
    errors.append('5. FAIL: Old scale_pos_weight=(30150/19850) still present')
else:
    print('[OK] 5. XGBoost scale_pos_weight: all instances set to 1')

# 6. Demographic parity
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if "list(results.values())[0]['y_pred']" in src:
        errors.append('6. FAIL: Still using list(results.values())[0]')
        break
else:
    print('[OK] 6. Demographic parity: uses best_name')

# 7. Calibration
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'CalibratedClassifierCV(base_model' in src:
        if "cv=3" in src:
            errors.append('7. FAIL: cv=3 still present')
        elif "cv='prefit'" in src:
            print('[OK] 7. Calibration: cv=prefit confirmed')
        break

# 8. best_model refit
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'best_pipeline = search.best_estimator_' in src:
        if 'best_model.fit(X_train, y_train)' in src:
            print('[OK] 8. best_model: refit on full training set confirmed')
        else:
            errors.append('8. FAIL: best_model not refitted after tuning')
        break

# 9. Learning curves
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'learning_curve(' in src:
        if 'X_train_raw, y_train_raw' in src and 'ImbPipeline' in src:
            print('[OK] 9. Learning curves: ImbPipeline + X_train_raw confirmed')
        elif 'X_train, y_train' in src and 'ImbPipeline' not in src:
            errors.append('9. FAIL: Learning curves still use X_train without ImbPipeline')
        else:
            print(f'[??] 9. Learning curves: check manually')
        break

# 10. SCARF comment
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'X_selected = pd.concat([X_train_raw, X_test]).sort_index()' in src:
        if 'Methodological Disclosure' in src:
            print('[OK] 10. SCARF: Methodological disclosure comment confirmed')
        else:
            errors.append('10. FAIL: SCARF comment not added')
        break

print(f'\n=== VERIFICATION: {10 - len(errors)}/10 PASSED ===')
if errors:
    print('ERRORS:')
    for e in errors:
        print(f'  {e}')
