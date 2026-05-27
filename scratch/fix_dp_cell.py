import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell containing 'cache_dp'
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'cache_dp' in source:
            # Let's replace 'pred' with 'y_pred' in the source code
            source_str = source.replace("results['FT-Transformer']['pred']", "results['FT-Transformer']['y_pred']")
            source_str = source_str.replace("results['Random Forest']['pred']", "results['Random Forest']['y_pred']")
            
            cell['source'] = [line + "\n" for line in source_str.split("\n")]
            if cell['source'][-1] == "\n":
                cell['source'][-1] = ""
                
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Successfully fixed KeyError in V5 notebook cell {idx}!")
            break
