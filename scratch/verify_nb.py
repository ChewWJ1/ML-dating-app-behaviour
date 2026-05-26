import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("Success: The V5 notebook parses perfectly as valid JSON.")
    print(f"Total cells in V5 notebook: {len(nb['cells'])}")
    
    code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    print(f"Code cells: {code_cells} | Markdown cells: {md_cells}")
    
    # Check if any cell has an empty source or malformed source
    malformed = 0
    for idx, c in enumerate(nb['cells']):
        if 'source' not in c or not isinstance(c['source'], list):
            print(f"Error: Malformed cell at index {idx}")
            malformed += 1
            
    if malformed == 0:
        print("Success: All cells are syntactically well-formed!")
except Exception as e:
    print(f"Error: Notebook verification failed! Details: {e}")
