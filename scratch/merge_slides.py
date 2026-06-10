import os
from pypdf import PdfWriter

slides_dir = "assets/slides"
output_pdf = "reports/Combined_Presentation_Slides.pdf"

# The three files in order, using all 10 pages of each
files_to_merge = [
    "Engineering_the_Dating_Pipeline.pdf",
    "The_Glass_Box_Protocol.pdf",
    "Proving_the_Null_Hypothesis.pdf"
]

writer = PdfWriter()

print("Merging PDFs:")
for f in files_to_merge:
    path = os.path.join(slides_dir, f)
    if os.path.exists(path):
        print(f"Adding {f}...")
        writer.append(path)
    else:
        print(f"Error: File not found - {path}")

os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
with open(output_pdf, "wb") as out:
    writer.write(out)

print(f"Successfully combined PDFs into: {output_pdf}")
print(f"Total pages: {len(writer.pages)}")
