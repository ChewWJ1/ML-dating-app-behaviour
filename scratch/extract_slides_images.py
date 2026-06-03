import os
from pypdf import PdfReader

slides_dir = "assets/slides"
output_dir = "scratch/extracted_slides"
os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(slides_dir) if f.endswith(".pdf")])

for f in files:
    path = os.path.join(slides_dir, f)
    reader = PdfReader(path)
    print(f"\nProcessing {f} ({len(reader.pages)} pages)")
    for idx in range(len(reader.pages)):
        page = reader.pages[idx]
        for image_file_object in page.images:
            img_name = f"{f[:-4]}_page_{idx+1}_{image_file_object.name}"
            img_path = os.path.join(output_dir, img_name)
            with open(img_path, "wb") as fp:
                fp.write(image_file_object.data)
            print(f"  Saved image: {img_name} ({len(image_file_object.data)} bytes)")
