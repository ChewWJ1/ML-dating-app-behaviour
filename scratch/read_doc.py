import docx

doc_path = r"reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx"
doc = docx.Document(doc_path)

out = []
for idx, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if not text:
        continue
    # Let's write paragraphs that contain keywords to inspect them
    keywords = ["Executive Summary", "14 baseline", "majority class", "1. Column Filtering", "StandardScaler", "Double Machine Learning", "Calibration", "References"]
    for kw in keywords:
        if kw in text:
            out.append(f"### Paragraph {idx} (Style: {p.style.name}) [KW: {kw}]\n\n{text}\n\n---\n")
            break

with open('scratch/inspect_docx.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

print("Successfully wrote inspected docx paragraphs to scratch/inspect_docx.md")
