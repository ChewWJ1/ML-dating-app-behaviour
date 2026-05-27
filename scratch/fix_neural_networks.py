import json
import os

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            
            # 1. Fix BayesianMLP cell
            if 'class BayesianMLP(nn.Module):' in src and '# Validate that preceding data' not in src:
                validation_code = (
                    "# Validate that preceding data variables are defined in the interactive session\n"
                    "if 'X_train' not in globals() or 'X_test' not in globals() or 'DEVICE' not in globals():\n"
                    "    raise NameError(\"❌ Required variables (X_train, X_test, DEVICE) are not defined in the active session.\\n\"\n"
                    "                    \"👉 Please run the preceding data loading and setup cells first.\")\n\n"
                )
                
                # Insert right after the imports in the cell
                if 'from sklearn.metrics import accuracy_score' in src:
                    parts = src.split('from sklearn.metrics import accuracy_score\n')
                    if len(parts) == 2:
                        new_src = parts[0] + 'from sklearn.metrics import accuracy_score\n\n' + validation_code + parts[1]
                        cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                          for i, line in enumerate(new_src.splitlines(True))]
                        modified = True
                        print("  - Added validation to BayesianMLP cell.")
            
            # 2. Fix FGSM Adv Attack cell
            if 'def fgsm_attack(' in src:
                new_src = src.replace('pytorch_model', 'bayesian_model')
                if new_src != src:
                    modified = True
                    print("  - Replaced pytorch_model with bayesian_model in FGSM cell.")
                    src = new_src
                
                if '# Validate that preceding neural network' not in src:
                    validation_code = (
                        "# Validate that preceding neural network variables are defined in the interactive session\n"
                        "if 'bayesian_model' not in globals() or 'X_test' not in globals() or 'DEVICE' not in globals():\n"
                        "    raise NameError(\"❌ Required variables (bayesian_model, X_test, DEVICE) are not defined in the active session.\\n\"\n"
                        "                    \"👉 Please run the preceding BayesianMLP cell first.\")\n\n"
                    )
                    
                    if 'import matplotlib.pyplot as plt\n' in src:
                        parts = src.split('import matplotlib.pyplot as plt\n')
                        if len(parts) == 2:
                            new_src = parts[0] + 'import matplotlib.pyplot as plt\n\n' + validation_code + parts[1]
                            cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                              for i, line in enumerate(new_src.splitlines(True))]
                            modified = True
                            print("  - Added validation to FGSM cell.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
