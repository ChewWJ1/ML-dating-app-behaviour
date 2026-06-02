import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "sv_2d" in source and "joblib.dump" in source and "cache_shap" in source:
            # We found the problematic cell!
            # Let's extract the lines and fix the order.
            lines = cell["source"]
            new_lines = []
            
            joblib_block = []
            in_joblib_block = False
            
            for line in lines:
                if "joblib.dump({" in line:
                    in_joblib_block = True
                    joblib_block.append(line)
                elif in_joblib_block:
                    joblib_block.append(line)
                    if "}, cache_shap)" in line:
                        in_joblib_block = False
                        # We do NOT append joblib_block to new_lines yet.
                else:
                    new_lines.append(line)
            
            # Now we need to insert the joblib_block AFTER sv_2d is initialized.
            # sv_2d is initialized around:
            # sv_2d = shap_values_obj.values
            # or sv_2d = shap_values_obj.values[1]
            # So we will insert it right before the plotting happens or at the very end of the cell.
            # Actually, let's insert it right before the first plotting command `shap.summary_plot`
            insert_idx = len(new_lines)
            for idx, line in enumerate(new_lines):
                if "shap.summary_plot(" in line or "mean_abs_shap =" in line or "plt.show()" in line:
                    insert_idx = idx
                    break
            
            # Insert the joblib_block
            new_lines = new_lines[:insert_idx] + joblib_block + new_lines[insert_idx:]
            
            cell["source"] = new_lines
            print("Fixed SHAP caching block order in notebook.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
