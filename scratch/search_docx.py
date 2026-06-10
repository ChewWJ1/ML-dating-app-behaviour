import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx')

print("=== PARAGRAPHS MENTIONING 'model' AND COUNT OR 'fold' OR 'student' ===")
for i, p in enumerate(doc.paragraphs):
    txt = p.text.lower()
    matches = []
    if any(term in txt for term in ["16 model", "14 model", "13 model", "15 model"]):
        matches.append("model-count")
    if any(term in txt for term in ["5-fold", "10-fold", "5 fold", "10 fold"]):
        matches.append("fold-count")
    if "student" in txt or "distill" in txt:
        matches.append("distill/student")
    
    if matches:
        print(f"Para {i} ({','.join(matches)}):")
        print(p.text)
        print("-" * 50)
