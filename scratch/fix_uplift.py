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
            
            if "pics_col_idx = list(X.columns).index('profile_pics_count')" in src and "T_train_raw = X_train[:, pics_col_idx]" in src:
                
                new_code = (
                    "    # Get treatment variable safely whether X_train is a DataFrame or Numpy Array\n"
                    "    if isinstance(X_train, pd.DataFrame):\n"
                    "        if 'profile_pics_count' in X_train.columns:\n"
                    "            T_train_raw = X_train['profile_pics_count'].values\n"
                    "        else:\n"
                    "            # If column names were lost but it still has original shape\n"
                    "            pics_col_idx = list(X.columns).index('profile_pics_count')\n"
                    "            T_train_raw = X_train.iloc[:, pics_col_idx].values\n"
                    "    else:\n"
                    "        if 'feature_names' in globals() and 'profile_pics_count' in feature_names:\n"
                    "            pics_col_idx = list(feature_names).index('profile_pics_count')\n"
                    "        else:\n"
                    "            pics_col_idx = list(X.columns).index('profile_pics_count')\n"
                    "        T_train_raw = X_train[:, pics_col_idx]\n"
                )
                
                pattern = r"    pics_col_idx = list\(X\.columns\)\.index\('profile_pics_count'\)\n    T_train_raw = X_train\[:, pics_col_idx\]\n"
                
                if re.search(pattern, src):
                    new_src = re.sub(pattern, new_code, src)
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                      for i, line in enumerate(new_src.splitlines(True))]
                    modified = True
                    print("  - Fixed Uplift Data Slicing Bug.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
