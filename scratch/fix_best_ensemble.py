import json

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            
            if 'teacher_probs_train = best_ensemble.predict_proba' in src and 'best_ensemble =' not in src:
                validation_code = (
                    "# Dynamically locate the best ensemble model to act as the teacher\n"
                    "best_ensemble = None\n"
                    "if 'tuned_results' in globals() and 'Random Forest' in tuned_results:\n"
                    "    best_ensemble = tuned_results['Random Forest']['model']\n"
                    "elif 'results' in globals() and 'Random Forest' in results:\n"
                    "    best_ensemble = results['Random Forest']['model']\n"
                    "elif 'best_model' in globals():\n"
                    "    best_ensemble = best_model\n"
                    "else:\n"
                    "    raise NameError(\"❌ Could not find a trained ensemble model (e.g., Random Forest) to act as the teacher.\\n\"\n"
                    "                    \"👉 Please run the model training and tuning cells first.\")\n\n"
                )
                
                # Insert it right before Step 1
                if '# Step 1: Get teacher\'s soft predictions' in src:
                    new_src = src.replace("# Step 1: Get teacher's soft predictions", validation_code + "# Step 1: Get teacher's soft predictions")
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                      for i, line in enumerate(new_src.splitlines(True))]
                    modified = True
                    print("  - Fixed NameError: injected best_ensemble definition.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
