import json

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            
            old_code = (
                "            # Convert X_train back to a dataframe with column names\n"
                "            X_train_df = pd.DataFrame(X_train, columns=X.columns)\n"
                "            train_df = X_train_df.copy()\n"
                "            train_df['target'] = y_train.values if hasattr(y_train, 'values') else y_train\n"
                "            \n"
                "            # Map features\n"
                "            d_dice = dice_ml.Data(dataframe=train_df, \n"
                "                                  continuous_features=list(numeric_cols), \n"
                "                                  outcome_name='target')"
            )
            
            new_code = (
                "            # Ensure X_train is a dataframe with correct active columns\n"
                "            if isinstance(X_train, pd.DataFrame):\n"
                "                X_train_df = X_train.copy()\n"
                "            else:\n"
                "                cols = feature_names if 'feature_names' in globals() else X.columns\n"
                "                X_train_df = pd.DataFrame(X_train, columns=cols)\n"
                "                \n"
                "            train_df = X_train_df.copy()\n"
                "            train_df['target'] = y_train.values if hasattr(y_train, 'values') else y_train\n"
                "            \n"
                "            # Filter continuous_features to only those still present after feature selection\n"
                "            valid_numeric_cols = [c for c in numeric_cols if c in train_df.columns]\n"
                "            \n"
                "            # Map features\n"
                "            d_dice = dice_ml.Data(dataframe=train_df, \n"
                "                                  continuous_features=valid_numeric_cols, \n"
                "                                  outcome_name='target')"
            )
            
            if "X_train_df = pd.DataFrame(X_train, columns=X.columns)" in src:
                # We need to replace it flexibly in case spacing is slightly different
                import re
                # Use regex to find and replace the block
                pattern = r"            # Convert X_train back to a dataframe with column names.*?outcome_name='target'\)"
                if re.search(pattern, src, flags=re.DOTALL):
                    new_src = re.sub(pattern, new_code, src, flags=re.DOTALL)
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                      for i, line in enumerate(new_src.splitlines(True))]
                    modified = True
                    print("  - Fixed DiCE Dataframe construction and numeric columns mapping.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
