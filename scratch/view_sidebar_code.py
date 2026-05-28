import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

with open(theme_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Printing lines 435 to 490 of {theme_path}:")
for idx in range(434, min(len(lines), 490)):
    print(f"  {idx + 1}: {lines[idx].rstrip()}")
