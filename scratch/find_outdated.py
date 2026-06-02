import sys
import docx

sys.stdout.reconfigure(encoding='utf-8')

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
doc = docx.Document(doc_path)

outdated_keywords = ["Random Forest", "60.48", "60.30", "accuracy", "AUC", "0.50", "0.51"]

for i, p in enumerate(doc.paragraphs):
    text = p.text
    if any(k in text for k in outdated_keywords):
        print(f"[{i}] {text}")
