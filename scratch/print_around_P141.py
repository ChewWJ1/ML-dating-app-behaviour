import docx
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.2 (final).docx"
doc = docx.Document(doc_path)

for idx in range(136, 149):
    p = doc.paragraphs[idx]
    has_image = "Drawing" in p._p.xml
    print(f"P{idx}: text='{p.text.strip()}' has_image={has_image}")
