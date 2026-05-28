import json
import base64
import os
import re

notebook_path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb'
output_dir = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\v5_plots'
os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

image_count = 0
last_header = "plot"

def sanitize_filename(name):
    # remove special characters, replace spaces with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_').lower()
    return name[:50]

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        headers = re.findall(r'^#+\s+(.*)', source, re.MULTILINE)
        if headers:
            last_header = sanitize_filename(headers[-1])
            
    if cell['cell_type'] == 'code':
        # check if it's plotting something specific in the source
        source = "".join(cell.get('source', []))
        if "confusion_matrix" in source:
            last_header = "confusion_matrix_" + last_header
        elif "roc_curve" in source:
            last_header = "roc_curve_" + last_header
            
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'display_data':
                if 'image/png' in output.get('data', {}):
                    image_count += 1
                    b64_data = output['data']['image/png']
                    
                    filename = f"{image_count:02d}_{last_header}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "wb") as fh:
                        fh.write(base64.b64decode(b64_data))
                    
                    print(f"Saved: {filename}")

print(f"Successfully extracted {image_count} images to assets/plots/")
