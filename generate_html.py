import os

image_files = os.listdir(r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots')
image_files.sort()

html = """
    <!-- ==================== TAB: VISUALS ==================== -->
    <div id="visuals-tab" class="tab-content">
      <div>
        <div class="section-header">
          <span class="section-title">Notebook Visualizations Gallery</span>
        </div>
        <p style="color:var(--text-secondary); margin-bottom:20px; font-size:14px;">High-resolution static plots extracted directly from the ML pipeline notebook.</p>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
"""

for file in image_files:
    if file.endswith('.png'):
        name = file.replace('.png', '').split('_', 1)[1].replace('_', ' ').title()
        html += f"""
        <div class="card" style="padding:15px; display:flex; flex-direction:column; align-items:center;">
          <div class="card-title" style="margin-bottom:15px; text-align:center;">{name}</div>
          <img src="assets/plots/{file}" style="width:100%; max-width:800px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
        </div>
"""

html += """
      </div>
    </div>
"""

with open('visuals_html.txt', 'w') as f:
    f.write(html)
print("HTML generated.")
