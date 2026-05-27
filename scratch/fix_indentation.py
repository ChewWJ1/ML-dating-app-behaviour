import json

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            
            if 'if isinstance(X_test, pd.DataFrame):' in src:
                # We need to correctly indent the block
                lines = src.split('\n')
                for i in range(len(lines)):
                    if 'if isinstance(X_test, pd.DataFrame):' in lines[i]:
                        # The if statement has 16 spaces
                        if lines[i+1] == '                X_test_df = X_test.copy()':
                            lines[i+1] = '                    X_test_df = X_test.copy()'
                            modified = True
                        if lines[i+2] == '            else:':
                            lines[i+2] = '                else:'
                            modified = True
                        if lines[i+3] == "                cols = feature_names if 'feature_names' in globals() else X.columns":
                            lines[i+3] = "                    cols = feature_names if 'feature_names' in globals() else X.columns"
                            modified = True
                        if lines[i+4] == '                X_test_df = pd.DataFrame(X_test, columns=cols)':
                            lines[i+4] = '                    X_test_df = pd.DataFrame(X_test, columns=cols)'
                            modified = True
                
                if modified:
                    cell['source'] = [line + ('\n' if i < len(lines) - 1 else '') for i, line in enumerate(lines)]
                    print("  - Fixed IndentationError.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
