import pypdf

pdf_path = "reports/WIA1006_WID3006_Group Assignment_2526.pdf"
reader = pypdf.PdfReader(pdf_path)

out = []
out.append(f"# PDF File: {pdf_path}\n")
out.append(f"Total Pages: {len(reader.pages)}\n")

for i, page in enumerate(reader.pages):
    out.append(f"## Page {i+1}\n")
    out.append(page.extract_text())
    out.append("\n---\n")

with open('scratch/rubric_text.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

print("Successfully extracted rubric text to scratch/rubric_text.md")
