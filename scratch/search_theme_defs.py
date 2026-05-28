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

for idx, line in enumerate(lines):
    if line.strip().startswith("def "):
        print(f"Line {idx + 1}: {line.strip()}")
        # Print next 5 lines
        for j in range(idx + 1, min(len(lines), idx + 8)):
            print(f"  {j + 1}: {lines[j]}")
