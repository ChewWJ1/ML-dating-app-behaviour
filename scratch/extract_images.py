import json
import base64
import os
import datetime

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8_patched_v2.ipynb"
assets_base_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets"
new_dir_name = "extracted_figures_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
target_dir = os.path.join(assets_base_dir, new_dir_name)

os.makedirs(target_dir, exist_ok=True)

print(f"Extracting images to: {target_dir}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

img_count = 0
for cell_idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            for key, value in data.items():
                if key.startswith('image/'):
                    ext = key.split('/')[1]
                    img_data = value
                    
                    # If it's a list, join it
                    if isinstance(img_data, list):
                        img_data = "".join(img_data)
                    
                    try:
                        img_bytes = base64.b64decode(img_data)
                        img_count += 1
                        filename = f"figure_{img_count:03d}.{ext}"
                        filepath = os.path.join(target_dir, filename)
                        
                        with open(filepath, 'wb') as img_file:
                            img_file.write(img_bytes)
                        print(f"Saved {filename}")
                    except Exception as e:
                        print(f"Failed to decode image in cell {cell_idx}: {e}")

print(f"Done. Extracted {img_count} images.")
