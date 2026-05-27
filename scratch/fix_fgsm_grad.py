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
            
            if 'def fgsm_attack(' in src and 'X_adv.requires_grad = True' not in src:
                # Replace the buggy gradient creation
                buggy_line = "    X_adv = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)\n"
                fixed_lines = (
                    "    X_adv = torch.tensor(X, dtype=torch.float32).to(device)\n"
                    "    X_adv.requires_grad = True\n"
                    "    model.zero_grad()\n"
                )
                
                if buggy_line in src:
                    new_src = src.replace(buggy_line, fixed_lines)
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') 
                                      for i, line in enumerate(new_src.splitlines(True))]
                    modified = True
                    print("  - Fixed FGSM gradient computation.")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved modified {file_path}")
    else:
        print("  - No changes needed.")

if __name__ == '__main__':
    fix_notebook('notebooks/ML_dating_app_behaviour V4.ipynb')
    fix_notebook('notebooks/ML_dating_app_behaviour V5.ipynb')
