import zipfile
import hashlib
import os

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.2 (final).docx"
extracted_plots_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\extracted_v5_plots"

def get_hash(data):
    return hashlib.md5(data).hexdigest()

# Get hashes of extracted plots
plot_hashes = {}
for filename in os.listdir(extracted_plots_dir):
    if filename.endswith(".png"):
        filepath = os.path.join(extracted_plots_dir, filename)
        with open(filepath, "rb") as f:
            content = f.read()
            plot_hashes[get_hash(content)] = filename
            print(f"Extracted plot: {filename} -> hash={get_hash(content)[:8]}, size={len(content)}")

print("\n--- Matching Zip Media with Extracted Plots ---")
mapping = {}
with zipfile.ZipFile(doc_path, 'r') as z:
    for name in sorted(z.namelist()):
        if name.startswith("word/media/"):
            content = z.read(name)
            h = get_hash(content)
            matched = plot_hashes.get(h, "None")
            print(f"  {name} -> matched={matched}, hash={h[:8]}, size={len(content)}")
            if matched != "None":
                mapping[name] = matched

import json
with open("scratch/zip_to_plot_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)
