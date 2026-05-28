import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

app_path = r"streamlit_app_v2/app.py"
if not os.path.exists(app_path):
    print("Error: app.py not found!")
    exit(1)

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

class_lines = []
for idx, line in enumerate(lines):
    if "um-header" in line or "header-container" in line or "header-button" in line or "nav-button" in line or "header_btn" in line:
        class_lines.append(idx)

for idx in class_lines:
    start = max(0, idx - 5)
    end = min(len(lines), idx + 20)
    print(f"\n--- Code Snippet around Line {idx + 1} ---")
    for j in range(start, end):
        print(f"{j + 1}: {lines[j].rstrip()}")
