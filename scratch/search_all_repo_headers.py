import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = "streamlit_app_v2"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            if "st.markdown" in code or "st.sidebar" in code or "st.html" in code or "components.html" in code:
                lines = code.split("\n")
                for idx, line in enumerate(lines):
                    if ("div" in line or "class=" in line or "header" in line or "button" in line) and ("st.markdown" in line or "st.sidebar" in line or "st.write" in line or "html" in line):
                        print(f"File: {path} | Line {idx + 1}: {line.strip()}")
                        # Print next 10 lines
                        for j in range(idx + 1, min(len(lines), idx + 12)):
                            print(f"  {j + 1}: {lines[j]}")
