import os
import sys

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    from pypdf import PdfReader, PdfWriter

def combine_pdfs():
    base_dir = r"C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\slides"
    
    slides_to_extract = [
        ("Tying_the_Data_Knot.pdf", 0), # Page 1
        ("Engineering_Digital_Romance.pdf", 3), # Page 4
        ("Proving_the_Null_Hypothesis.pdf", 2), # Page 3
        ("Proving_the_Null_Hypothesis.pdf", 5), # Page 6
        ("Proving_the_Null_Hypothesis.pdf", 4), # Page 5
        ("SwipeIQ_V2_Strategic_Roadmap.pdf", 3), # Page 4
        ("Proving_the_Null_Hypothesis.pdf", 7), # Page 8
        ("SwipeIQ_V2_Strategic_Roadmap.pdf", 4)  # Page 5
    ]
    
    writer = PdfWriter()
    
    for filename, page_num in slides_to_extract:
        filepath = os.path.join(base_dir, filename)
        reader = PdfReader(filepath)
        page = reader.pages[page_num]
        writer.add_page(page)
        
    output_path = os.path.join(base_dir, "Final_Presentation_Slides.pdf")
    with open(output_path, "wb") as f:
        writer.write(f)
        
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    combine_pdfs()
