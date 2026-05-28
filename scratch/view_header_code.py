import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

with open(theme_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines in theme.py: {len(lines)}")

# Focus on lines 490 to 762 (0-indexed: 489 to 761)
start_line = 489
end_line = 761

print(f"\n--- Scanning render_header in theme.py (Lines {start_line+1} to {end_line+1}) ---")
for idx in range(start_line, end_line):
    line = lines[idx]
    # Check for links, classes, buttons, onclicks
    if any(x in line.lower() for x in ["href", "click", "button", "logo", "menu", "dropdown", "nav-item"]):
        print(f"Line {idx + 1}: {line.strip()}")
