import fitz  # PyMuPDF
import os

pdf_path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\slides\Absolute_Best_Presentation_Slides_21.pdf'
output_dir = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\slides\rendered_pages'
os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")

for i, page in enumerate(doc):
    # Render at 2x resolution for readability
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(output_dir, f"page_{i+1:02d}.png")
    pix.save(out_path)
    print(f"Rendered page {i+1} -> {out_path}")

doc.close()
print("\nDone!")
