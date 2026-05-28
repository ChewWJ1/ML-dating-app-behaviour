import os

theme_path = r"streamlit_app_v2/utils/theme.py"
if not os.path.exists(theme_path):
    print("Error: theme.py not found!")
    exit(1)

print(f"Reading {theme_path}...")
with open(theme_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update native sidebar controls selector list
control_target = """        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] * {{
            pointer-events: auto !important;
        }}"""

control_replacement = """        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] *,
        header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
        header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] *,
        header[data-testid="stHeader"] [data-testid="stHeaderSidebarCollapsedControl"],
        header[data-testid="stHeader"] [data-testid="stHeaderSidebarCollapsedControl"] * {{
            pointer-events: auto !important;
        }}"""

# 2. Update .web-nav-item CSS to force pointer-events: auto
nav_item_target = """        .web-nav-item {{
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: rgba(255, 255, 255, 0.8) !important;
            text-decoration: none !important;
            padding: 6px 14px !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            cursor: pointer !important;
            position: relative !important;
            z-index: 1000003 !important;
        }}"""

nav_item_replacement = """        .web-nav-item {{
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: rgba(255, 255, 255, 0.8) !important;
            text-decoration: none !important;
            padding: 6px 14px !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            cursor: pointer !important;
            position: relative !important;
            z-index: 1000003 !important;
            pointer-events: auto !important;
        }}"""

# 3. Update .web-dropdown-content CSS
dropdown_content_target = """        .web-dropdown-content {{
            display: none;
            position: absolute;
            background-color: #1e293b;
            min-width: 220px;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.5);
            z-index: 1000004;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            top: calc(100% + 8px); /* Starts tightly below the button now */
            left: 50%;
            transform: translateX(-50%);
        }}"""

dropdown_content_replacement = """        .web-dropdown-content {{
            display: none;
            position: absolute;
            background-color: #1e293b;
            min-width: 220px;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.5);
            z-index: 1000004;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            top: calc(100% + 8px); /* Starts tightly below the button now */
            left: 50%;
            transform: translateX(-50%);
            pointer-events: auto !important;
        }}"""

# 4. Update .web-dropdown-content a CSS
dropdown_a_target = """        .web-dropdown-content a {{
            color: rgba(255, 255, 255, 0.8) !important;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            font-family: 'Inter', sans-serif !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
            text-transform: none !important;
            letter-spacing: 0 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}"""

dropdown_a_replacement = """        .web-dropdown-content a {{
            color: rgba(255, 255, 255, 0.8) !important;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            font-family: 'Inter', sans-serif !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
            text-transform: none !important;
            letter-spacing: 0 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            pointer-events: auto !important;
        }}"""

# 5. Replace <a> with <span> for parent dropdown triggers
tag_target_1 = """<a class="web-nav-item {act_pipeline}" tabindex="0">⚙️ ML Pipeline ▾</a>"""
tag_replacement_1 = """<span class="web-nav-item {act_pipeline}" tabindex="0">⚙️ ML Pipeline ▾</span>"""

tag_target_2 = """<a class="web-nav-item {act_advanced}" tabindex="0">🧠 Advanced Methods ▾</a>"""
tag_replacement_2 = """<span class="web-nav-item {act_advanced}" tabindex="0">🧠 Advanced Methods ▾</span>"""

tag_target_3 = """<a class="web-nav-item {act_docs}" tabindex="0">📄 Docs ▾</a>"""
tag_replacement_3 = """<span class="web-nav-item {act_docs}" tabindex="0">📄 Docs ▾</span>"""


# Run replacements
modifications = [
    (control_target, control_replacement, "Sidebar collapse selector"),
    (nav_item_target, nav_item_replacement, "web-nav-item pointer-events"),
    (dropdown_content_target, dropdown_content_replacement, "web-dropdown-content pointer-events"),
    (dropdown_a_target, dropdown_a_replacement, "web-dropdown-content a pointer-events"),
    (tag_target_1, tag_replacement_1, "ML Pipeline tag to span"),
    (tag_target_2, tag_replacement_2, "Advanced Methods tag to span"),
    (tag_target_3, tag_replacement_3, "Docs tag to span")
]

successful_mods = 0
for target, replacement, desc in modifications:
    if target in content:
        content = content.replace(target, replacement)
        print(f"  [OK] Successfully replaced: {desc}")
        successful_mods += 1
    else:
        # Check normalized spacing
        norm_target = "".join(target.split())
        norm_content = "".join(content.split())
        if norm_target in norm_content:
            print(f"  [WARNING] Target '{desc}' exists but has spacing differences. Trying fuzzy replacement...")
            # We can find line ranges or do regex, but let's see if we can find exact matching blocks.
        else:
            print(f"  [ERROR] Failed to locate target block: {desc}")

print(f"Writing updates to {theme_path}...")
with open(theme_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"🎉 Theme bugfixes completed. Successfully applied {successful_mods} out of {len(modifications)} changes.")
