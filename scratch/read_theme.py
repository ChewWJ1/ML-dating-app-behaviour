import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

with open(theme_path, "r", encoding="utf-8") as f:
    code = f.read()

lines = code.split("\n")
print(f"Total lines in theme.py: {len(lines)}")

# Search for um-header or HTML code inside theme.py
for idx, line in enumerate(lines):
    if "um-" in line or "header" in line.lower() or "button" in line.lower() or "navbar" in line.lower() or "href" in line.lower() or "<div" in line:
        # Check if it has a definition
        if "def " in line or "st." in line or "class=" in line or "background" in line or "css" in line.lower():
            print(f"Line {idx + 1}: {line.strip()}")
            # Print a snippet of next 15 lines
            for j in range(idx + 1, min(len(lines), idx + 18)):
                print(f"  {j + 1}: {lines[j]}")
