import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

with open(theme_path, "r", encoding="utf-8") as f:
    code = f.read()

# Verify some key replacements exist in the file
verifications = [
    "stSidebarCollapseButton",
    "stHeaderSidebarCollapsedControl",
    "pointer-events: auto !important;\n        }", # web-nav-item pointer-events
    "label class=\"web-nav-item {act_pipeline}\""
]

print("Checking theme.py for modifications:")
all_ok = True
for v in verifications:
    if v in code:
        print(f"  [OK] Found patch element: '{v}'")
    else:
        print(f"  [ERROR] Missing patch element: '{v}'")
        all_ok = False

if all_ok:
    print("\n🎉 Verification passed! All bugfixes successfully applied.")
    # Check syntax by compiling
    try:
        compile(code, theme_path, "exec")
        print("  [OK] Python syntax compiles perfectly.")
    except Exception as e:
        print(f"  [ERROR] Syntax compilation failed: {e}")
else:
    print("\n❌ Verification failed.")
