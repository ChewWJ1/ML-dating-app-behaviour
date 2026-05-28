import os

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

with open(theme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Locate padding in .website-header class block
target_padding = "            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;\n            padding: 0 24px !important;\n            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;"
replacement_padding = "            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;\n            padding: 0 24px 0 80px !important; /* Leaves 80px empty space on left for sidebar toggle */\n            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;"

if target_padding in content:
    content = content.replace(target_padding, replacement_padding)
    print("  [OK] Successfully replaced header padding.")
else:
    print("  [ERROR] Failed to locate padding selector block!")

with open(theme_path, "w", encoding="utf-8") as f:
    f.write(content)
