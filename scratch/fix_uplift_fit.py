import json
import re

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            
            new_src = src
            # Fix X_train[idx_treat]
            if "X_train[idx_treat]" in new_src:
                new_src = new_src.replace(
                    "X_train[idx_treat]", 
                    "(X_train.iloc[idx_treat] if isinstance(X_train, pd.DataFrame) else X_train[idx_treat])"
                )
                modified = True
            
            # Fix X_train[idx_ctrl]
            if "X_train[idx_ctrl]" in new_src:
                new_src = new_src.replace(
                    "X_train[idx_ctrl]", 
                    "(X_train.iloc[idx_ctrl] if isinstance(X_train, pd.DataFrame) else X_train[idx_ctrl])"
                )
                modified = True
                
            if modified:
                cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                  for i, line in enumerate(new_src.splitlines(True))]
                print("  - Fixed Uplift fit data slicing bug.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
