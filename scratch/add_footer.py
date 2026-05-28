import os
import glob
import sys

# Set standard output encoding to UTF-8 to prevent any Windows console crashes
if sys.stdout.encoding != 'utf-8':
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pages_dir = os.path.join(ROOT_DIR, "streamlit_app_v2", "pages")
main_app = os.path.join(ROOT_DIR, "streamlit_app_v2", "app.py")

# Find all python script files in pages and app
files = glob.glob(os.path.join(pages_dir, "*.py"))
files.append(main_app)

footer_block = """

# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
"""

print(f"Total files detected: {len(files)}")

for filepath in files:
    if "__init__" in filepath or "pycache" in filepath:
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if footer render call already exists to prevent duplicate appends
        if "render_footer()" not in content:
            # Safely print filename by replacing non-ASCII characters if needed
            safe_name = os.path.basename(filepath).encode('ascii', errors='replace').decode('ascii')
            print(f"Adding footer to: {safe_name}")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(footer_block)
        else:
            safe_name = os.path.basename(filepath).encode('ascii', errors='replace').decode('ascii')
            print(f"Footer already exists in: {safe_name}")
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

print("Footer injection completed successfully!")
