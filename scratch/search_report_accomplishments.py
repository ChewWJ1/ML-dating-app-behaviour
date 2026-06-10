import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx')

print("=== SEARCH PARAGRAPHS FOR 'accomplishments' OR 'checkpoint' ===")
for i, p in enumerate(doc.paragraphs):
    txt = p.text.lower()
    if "accomplishments" in txt or "checkpoint" in txt or "cv_results" in txt:
        print(f"Para {i}: {p.text}")
        print("-" * 50)
