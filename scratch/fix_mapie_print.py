import json

def inject_print(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'coverage = np.mean(classification_coverage_score(y_test_conformal, y_sets[:, :, i]))' in src and 'print("SHAPES:' not in src:
                new_src = src.replace(
                    "coverage = np.mean(classification_coverage_score(y_test_conformal, y_sets[:, :, i]))",
                    'print(f"SHAPES: y_test_conformal: {getattr(y_test_conformal, \'shape\', len(y_test_conformal))}, y_sets: {y_sets[:, :, i].shape}")\n    coverage = np.mean(classification_coverage_score(y_test_conformal, y_sets[:, :, i]))'
                )
                cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                  for i, line in enumerate(new_src.splitlines(True))]
                modified = True
                print("  - Injected print statement.")
                
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")

if __name__ == '__main__':
    inject_print('notebooks/ML_dating_app_behaviour V4.ipynb')
    inject_print('notebooks/ML_dating_app_behaviour V5.ipynb')
