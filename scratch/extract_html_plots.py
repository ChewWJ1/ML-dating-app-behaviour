import re
import os
import base64
from bs4 import BeautifulSoup

html_path = "notebooks/exports/ML_dating_app_behaviour V5.html"
output_dir = "assets/notebook_plots_extracted"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(html_path):
    print(f"HTML file not found at {html_path}")
    exit(1)

print(f"Reading HTML file: {html_path}...")
with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

img_tags = soup.find_all("img")
print(f"Found {len(img_tags)} image tags in HTML.")

count = 0
for idx, img in enumerate(img_tags):
    src = img.get("src", "")
    if src.startswith("data:image/png;base64,"):
        base64_data = src.replace("data:image/png;base64,", "")
        image_data = base64.b64decode(base64_data)
        
        # We can look for the nearest cell heading to give it a descriptive name
        # For now, let's save as plot_{idx}.png
        img_name = f"plot_{idx+1:02d}.png"
        img_path = os.path.join(output_dir, img_name)
        with open(img_path, "wb") as f_img:
            f_img.write(image_data)
        
        # Let's try to find text context around it
        parent = img.parent
        context_text = ""
        for i in range(3):
            if parent:
                # Get some text sibling or parent text
                text_content = parent.get_text()[:100].strip().replace('\n', ' ')
                if text_content:
                    context_text = text_content
                    break
                parent = parent.parent
                
        # Safe print for Windows console
        safe_context = context_text.encode('ascii', 'ignore').decode('ascii')
        print(f"Saved {img_path} ({len(image_data)} bytes) - Context: {safe_context[:80]}")
        count += 1

print(f"Successfully extracted {count} plots from HTML to {output_dir}")
