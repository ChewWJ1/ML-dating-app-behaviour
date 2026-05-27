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
            
            # Look for the exact block
            pattern = (
                r"if os\.path\.exists\(cache_mapie\):\n"
                r"    print\(\"⏭️  Loading cached MAPIE Conformal Prediction sets\.\.\.\"\)\n"
                r"    mapie_data = joblib\.load\(cache_mapie\)\n"
                r"    mapie = mapie_data\.get\('mapie', None\)\n"
                r"    y_pred = mapie_data\['y_pred'\]\n"
                r"    y_sets = mapie_data\['y_sets'\]\n"
                r"    alpha_levels = mapie_data\['alpha_levels'\]\n"
                r"else:"
            )
            
            new_code = (
                "mapie_data = joblib.load(cache_mapie) if os.path.exists(cache_mapie) else None\n"
                "if mapie_data is not None and len(mapie_data.get('y_sets', [])) == len(y_test_conformal):\n"
                "    print(\"⏭️  Loading cached MAPIE Conformal Prediction sets...\")\n"
                "    mapie = mapie_data.get('mapie', None)\n"
                "    y_pred = mapie_data['y_pred']\n"
                "    y_sets = mapie_data['y_sets']\n"
                "    alpha_levels = mapie_data['alpha_levels']\n"
                "else:\n"
                "    if mapie_data is not None:\n"
                "        print(\"⚠️ Cached MAPIE data shape mismatch (due to different test splits/features). Recomputing...\")\n"
            )
            
            if re.search(pattern, src):
                new_src = re.sub(pattern, new_code, src)
                cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                  for i, line in enumerate(new_src.splitlines(True))]
                modified = True
                print("  - Fixed MAPIE cache length mismatch bug.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
