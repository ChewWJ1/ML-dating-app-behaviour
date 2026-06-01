import json
file_path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V7_Strict.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

capture = False
with open('metrics.txt', 'w', encoding='utf-8') as out_f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            if any(h in src for h in ['10.3 Model Comparison Table', '10.6 Classification Reports', '12.3 Before vs After', '12.4 Best Tuned Model', '13.2 Final Model Summary']):
                out_f.write(f"\n\n======================\n{src[:100]}\n======================\n")
                capture = True
            elif src.startswith('#'):
                capture = False
        elif cell['cell_type'] == 'code' and capture:
            for out in cell.get('outputs', []):
                if out.get('output_type') == 'stream':
                    out_f.write(''.join(out.get('text', [])) + '\n')
                elif 'data' in out:
                    if 'text/plain' in out['data']:
                        out_f.write(''.join(out['data']['text/plain']) + '\n')
