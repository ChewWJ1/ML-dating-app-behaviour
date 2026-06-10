import os
from pypdf import PdfReader

slides_dir = "assets/slides"
files = sorted([f for f in os.listdir(slides_dir) if f.endswith(".pdf")])

for f in files:
    path = os.path.join(slides_dir, f)
    reader = PdfReader(path)
    print(f"\nFile: {f}")
    print(f"Number of pages: {len(reader.pages)}")
    if len(reader.pages) > 0:
        page = reader.pages[0]
        print(f"Page 1 keys: {page.keys()}")
        if '/Resources' in page:
            resources = page['/Resources']
            print(f"  Resources keys: {resources.keys()}")
            if '/XObject' in resources:
                xobjects = resources['/XObject']
                print(f"  XObjects: {list(xobjects.keys())}")
