import re

with open('dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

with open('visuals_html.txt', 'r', encoding='utf-8') as f:
    visuals_html = f.read()

# 1. Inject sidebar nav
sidebar_nav = """  <span class="nav-section-label">System</span>

  <a class="nav-item" href="javascript:void(0)" onclick="switchTab('visuals')">
    <span class="nav-icon">🖼️</span>
    <span>Notebook Visuals</span>
  </a>
"""
content = content.replace('  <span class="nav-section-label">System</span>\n', sidebar_nav)

# 2. Inject javascript logic
js_logic = """    } else if (tabId === 'visuals') {
      headerTitle.innerText = "Notebook Visualizations Gallery";
      headerDesc.innerText = "Static high-resolution plots extracted from the original exploratory and ML pipelines";
"""
content = content.replace("    } else if (tabId === 'reports') {", js_logic + "    } else if (tabId === 'reports') {")

# 3. Inject html content before Reports tab
content = content.replace("    <!-- ==================== TAB: REPORTS ==================== -->", visuals_html + "\n    <!-- ==================== TAB: REPORTS ==================== -->")

with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Dashboard successfully injected with notebook visuals.")
