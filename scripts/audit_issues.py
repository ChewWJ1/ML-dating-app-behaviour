import json

with open(r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. API token
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'TABPFN_TOKEN' in src and 'eyJ' in src:
        print(f'=== ISSUE 1: API TOKEN (Cell {i}) ===')
        for line in src.split('\n'):
            if 'TABPFN_TOKEN' in line or 'HF_TOKEN' in line:
                print(f'  {line[:120]}')

# 2. StandardScaler print
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'StandardScaler' in src:
        print(f'=== ISSUE 2: StandardScaler print (Cell {i}) ===')
        for line in src.split('\n'):
            if 'StandardScaler' in line:
                print(f'  {line}')

# 3. Duplicate param_grids
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'param_grids' in src and 'LightGBM' in src and 'CatBoost' in src:
        print(f'=== ISSUE 3: param_grids (Cell {i}) ===')
        count_lgb = src.count("'LightGBM'")
        count_cat = src.count("'CatBoost'")
        print(f'  LightGBM key count: {count_lgb}, CatBoost key count: {count_cat}')

# 4. DML in-sample prediction
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'model_T.fit' in src and 'predict_proba' in src:
        print(f'=== ISSUE 4: DML in-sample (Cell {i}) ===')
        for line in src.split('\n'):
            if 'model_T' in line or 'model_Y' in line or 'predict_proba' in line or 'predict(' in line:
                print(f'  {line}')

# 5. XGBoost scale_pos_weight
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'scale_pos_weight' in src:
        print(f'=== ISSUE 5: scale_pos_weight (Cell {i}) ===')
        for line in src.split('\n'):
            if 'scale_pos_weight' in line:
                print(f'  {line}')

# 6. Demographic parity
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'list(results.values())[0]' in src:
        print(f'=== ISSUE 6: Demographic parity (Cell {i}) ===')
        for line in src.split('\n'):
            if 'list(results' in line:
                print(f'  {line}')

# 7. Calibration cv=3
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'CalibratedClassifierCV' in src:
        print(f'=== ISSUE 7: Calibration (Cell {i}) ===')
        for line in src.split('\n'):
            if 'CalibratedClassifier' in line:
                print(f'  {line}')

# 8. best_model extraction after tuning
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'best_pipeline.named_steps' in src:
        print(f'=== ISSUE 8: best_model extraction (Cell {i}) ===')
        for line in src.split('\n'):
            if 'best_pipeline' in line or 'best_model' in line:
                print(f'  {line}')

# 9. Learning curves
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'learning_curve(' in src:
        print(f'=== ISSUE 9: Learning curves (Cell {i}) ===')
        for line in src.split('\n'):
            if 'learning_curve' in line:
                print(f'  {line}')

# 10. SCARF comment
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'X_selected = pd.concat' in src:
        print(f'=== ISSUE 10: SCARF X_selected (Cell {i}) ===')
        for line in src.split('\n'):
            if 'X_selected' in line and 'concat' in line:
                print(f'  {line}')
