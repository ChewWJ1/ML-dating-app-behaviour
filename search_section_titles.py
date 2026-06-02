import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx')

print("=== HEADINGS OR PARAGRAPHS CONTAINING 10.3 OR 10.7 OR 9. OR 17. ===")
for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    if not txt:
        continue
    txt_lower = txt.lower()
    if any(x in txt_lower for x in ["10.3", "10.7", "section 9", "section 17"]):
        print(f"Para {i}: {txt}")
        print("-" * 50)
    # Check if it starts with number and has model
    if txt.startswith(("9.", "10.3", "10.7", "17.")) or "10.3" in txt or "10.7" in txt:
        print(f"Para {i} (starts with section number): {txt}")
        print("-" * 50)
