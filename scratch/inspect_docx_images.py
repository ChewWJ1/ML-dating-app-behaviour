import docx

doc_path = "reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5 SOTA.docx"
doc = docx.Document(doc_path)

inline_shapes = doc.inline_shapes
print(f"Total inline shapes (images): {len(inline_shapes)}")

# Check for image placements or references
for idx, shape in enumerate(inline_shapes):
    print(f"Shape {idx}: Type={shape.type}, Width={shape.width}, Height={shape.height}")

# Let's also look for text that references figures or images
for idx, p in enumerate(doc.paragraphs):
    if "Figure" in p.text or "figure" in p.text.lower():
        print(f"P {idx}: {p.text[:120]}...")
