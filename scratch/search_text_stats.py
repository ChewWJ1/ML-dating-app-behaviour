import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/docx_full_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for sentences containing "feature", "PCA", "dimension", "t-test", "DML", "Average Treatment Effect", "conformal"
keywords = ["feature", "dimension", "PCA", "t-test", "DML", "ATE", "Treatment Effect", "conformal"]

lines = text.split("\n")
print(f"Total lines in docx_full_text.txt: {len(lines)}")

for kw in keywords:
    print(f"\nMatches for keyword '{kw}':")
    count = 0
    for line in lines:
        if re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE):
            # Print paragraph number and text
            print(f"  {line[:150]}...")
            count += 1
            if count >= 10:
                print("  ... truncated")
                break
