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
            
            new_code = (
                "            if isinstance(X_test, pd.DataFrame):\n"
                "                X_test_df = X_test.copy()\n"
                "            else:\n"
                "                cols = feature_names if 'feature_names' in globals() else X.columns\n"
                "                X_test_df = pd.DataFrame(X_test, columns=cols)\n"
            )
            
            if "X_test_df = pd.DataFrame(X_test, columns=X.columns)" in src:
                # Use regex to replace
                pattern = r"            X_test_df = pd\.DataFrame\(X_test, columns=X\.columns\)\n"
                if re.search(pattern, src):
                    new_src = re.sub(pattern, new_code, src)
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                      for i, line in enumerate(new_src.splitlines(True))]
                    modified = True
                    print("  - Fixed DiCE X_test Dataframe construction.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
