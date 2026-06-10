import json

with open('c:/Users/HP/Documents/GitHub/ML-dating-app-behaviour/notebooks/ML_dating_app_behaviour V8_patched_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

out = []
for i, c in enumerate(nb.get('cells', [])):
    if c['cell_type'] == 'code' and 'outputs' in c:
        out.append(f'\n--- Cell {i} Outputs ---')
        for o in c['outputs']:
            if 'text' in o:
                out.append(''.join(o['text']))
            elif 'data' in o and 'text/plain' in o['data']:
                out.append(''.join(o['data']['text/plain']))

with open('c:/Users/HP/Documents/GitHub/ML-dating-app-behaviour/extracted_outputs.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
