import docx
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
doc = docx.Document(doc_path)

print(f"Searching for model names in report paragraphs...")

keywords = [r"XGBoost", r"LightGBM", r"CatBoost", r"Random Forest"]

for idx, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if not text:
        continue
    for kw in keywords:
        if re.search(r'\b' + kw + r'\b', text, re.IGNORECASE):
            print(f"\nP{idx} (Style: {p.style.name}):")
            print(f"  {text[:200]}...")
            break
