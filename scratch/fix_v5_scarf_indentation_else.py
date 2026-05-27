import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_found = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = "".join(source)
        
        # Check if this is the SCARF cell
        if "cache_scarf = '../models_v5/scarf.joblib'" in source_str and "# === FINE-TUNING PHASE: Freeze encoder, train classifier head ===" in source_str:
            print("Found SCARF cell!")
            new_source = []
            
            # We want to identify the fine-tuning block and indent it by 4 spaces
            # The block starts with "# === FINE-TUNING PHASE:"
            # and ends before "# Train any downstream classifier"
            
            in_fine_tuning = False
            for line in source:
                if "# === FINE-TUNING PHASE: Freeze encoder, train classifier head ===" in line:
                    in_fine_tuning = True
                elif "# Train any downstream classifier on the LEARNED REPRESENTATIONS" in line:
                    in_fine_tuning = False
                
                if in_fine_tuning:
                    # Indent the line by 4 spaces if it is not empty
                    if line.strip():
                        new_source.append("    " + line)
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            target_found = True
            break

if target_found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully moved SCARF fine-tuning block inside the else block!")
else:
    print("SCARF cell not found!")
