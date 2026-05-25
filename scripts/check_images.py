import json

notebook_path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\ML_dating_app_behaviour.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

image_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'display_data':
                if 'image/png' in output.get('data', {}):
                    image_count += 1

print(f"Found {image_count} PNG images in the notebook.")
