import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx')

print("=== DETAILED SEARCH IN SECTIONS 9, 10, 16, 17 ===")
for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    if not txt:
        continue
    # Let's print paragraphs with index so we can see their exact wording and styles
    txt_lower = txt.lower()
    # Check if this paragraph is in or near the targets
    # Let's print if it mentions numbers 16 or 14 and model/architecture/estimator, or contains "Friedman", or "Student"
    if any(x in txt_lower for x in ["16 model", "14 model", "16 diverse", "14 custom", "friedman", "student", "distill"]):
        print(f"Para {i}: {txt}")
        print("-" * 50)
    elif "16" in txt or "14" in txt or "13" in txt:
        if any(w in txt_lower for w in ["model", "architecture", "algorithm", "classifier", "estimator"]):
            print(f"Para {i} (contains count + model): {txt}")
            print("-" * 50)
