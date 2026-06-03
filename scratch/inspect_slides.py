import os
from pypdf import PdfReader

slides_dir = "assets/slides"
files = sorted([f for f in os.listdir(slides_dir) if f.endswith(".pdf")])

for f in files:
    path = os.path.join(slides_dir, f)
    print(f"\n==================== {f} ====================")
    try:
        reader = PdfReader(path)
        print("Metadata:")
        meta = reader.metadata
        if meta:
            for k, v in meta.items():
                print(f"  {k}: {v}")
        else:
            print("  No metadata")
            
        print("Outlines:")
        outlines = reader.outline
        if outlines:
            print(f"  Found outline of length: {len(outlines)}")
            # Recursively print outlines if needed, but let's print representation
            print(f"  Outline: {outlines}")
        else:
            print("  No outline")
    except Exception as e:
        print(f"  Error: {e}")
