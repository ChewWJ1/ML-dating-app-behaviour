import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

mod_cache = 0
mod_reports = 0

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        cell_changed = False
        
        for line in source:
            new_line = line
            
            # 1. Replace cache folder
            if "models_v4_cache" in line:
                new_line = new_line.replace("models_v4_cache", "../models_v5")
                mod_cache += 1
                cell_changed = True
            
            # 2. Replace reports folder
            target_path_def = "reports_dir = '../reports' if os.path.exists('../reports') or not os.path.exists('reports') else 'reports'"
            new_path_def = "reports_dir = '../assets/notebook_plots' if os.path.exists('../assets') or not os.path.exists('assets') else 'assets/notebook_plots'"
            if target_path_def in line:
                new_line = new_line.replace(target_path_def, new_path_def)
                mod_reports += 1
                cell_changed = True
            
            new_source.append(new_line)
        
        if cell_changed:
            cell['source'] = new_source

if mod_cache > 0 or mod_reports > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Redirection complete! Replaced {mod_cache} cache instances and {mod_reports} report path definitions.")
else:
    print("No matches found to replace!")
