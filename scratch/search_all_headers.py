import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

app_path = r"streamlit_app_v2/app.py"
if not os.path.exists(app_path):
    print("Error: app.py not found!")
    exit(1)

with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

lines = code.split("\n")
print(f"Total lines in app.py: {len(lines)}")

# Search for st.markdown calls containing custom divs or logos or links
for idx, line in enumerate(lines):
    if "st.markdown" in line and ("div" in line or "class" in line or "a href" in line or "button" in line):
        print(f"\nLine {idx + 1}: {line.strip()}")
        # Print next 15 lines
        for j in range(idx + 1, min(len(lines), idx + 20)):
            print(f"  {j + 1}: {lines[j]}")
