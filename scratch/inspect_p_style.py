import docx
from docx import Document

doc = Document("reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx")
p = doc.paragraphs[216]
print("Paragraph Text:", p.text)
print("Paragraph Alignment:", p.alignment)
print("Space After:", p.paragraph_format.space_after)
print("Space Before:", p.paragraph_format.space_before)
print("Line Spacing:", p.paragraph_format.line_spacing)
print("Runs count:", len(p.runs))
for idx, r in enumerate(p.runs):
    print(f"Run {idx}: '{r.text}'")
    print(f"  Font Name: {r.font.name}")
    print(f"  Font Size: {r.font.size}")
    print(f"  Bold: {r.font.bold}")
    print(f"  Italic: {r.font.italic}")
    print(f"  Color: {r.font.color.rgb}")
