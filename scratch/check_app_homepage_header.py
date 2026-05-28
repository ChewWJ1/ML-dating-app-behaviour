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

# Look for header rendering or imports of theme utilities
print("Searching for render_header or st.markdown in app.py:")
for idx, line in enumerate(lines):
    if "render_header" in line or "theme.py" in line or "inject_css" in line:
        print(f"  Line {idx + 1}: {line.strip()}")
        # Print next 5 lines
        for j in range(idx + 1, min(len(lines), idx + 8)):
            print(f"    {j + 1}: {lines[j]}")
