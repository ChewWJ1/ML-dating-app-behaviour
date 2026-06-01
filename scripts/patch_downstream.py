import json
with open(r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    
    # Fix Calibration cell
    if 'champion_name = best_name' in src and 'for name in [' in src:
        new_src = src.replace("champion_name = best_name\nfor name in ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'Random Forest', 'XGBoost', 'LightGBM']:\n    if name in results:\n        champion_name = name\n        break", 'champion_name = best_name')
        new_src = new_src.replace("r_entry = results[champion_name]", "r_entry = tuned_results.get(champion_name) or results.get(champion_name)")
        new_src = new_src.replace("if champion_name and champion_name in results:", "target_dict = tuned_results if champion_name in tuned_results else results\nif champion_name:")
        new_src = new_src.replace("results[champion_name]", "target_dict[champion_name]")
        
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        if cell['source']: cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('✅ Patched Calibration')

    # Fix Conformal Prediction
    if "if 'best_model' not in globals() or best_model is None:" in src and 't_res =' in src:
        new_src = src.replace("    for name in ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'Random Forest', 'XGBoost', 'LightGBM']:\n        if name in t_res:\n            best_model = t_res[name].get('model')\n            print(f\"👉 Resolved best_model from tuned_results: {name}\")\n            break\n        elif name in b_res:\n            best_model = b_res[name].get('model')\n            print(f\"👉 Resolved best_model from baseline results: {name}\")\n            break", """    if best_name in t_res:
        best_model = t_res[best_name].get('model')
        print(f"👉 Dynamically resolved best_model from tuned_results: {best_name}")
    elif best_name in b_res:
        best_model = b_res[best_name].get('model')
        print(f"👉 Dynamically resolved best_model from baseline results: {best_name}")""")
        
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        if cell['source']: cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('✅ Patched Conformal Prediction')

with open(r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
