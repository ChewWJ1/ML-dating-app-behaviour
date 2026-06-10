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
        # Speaker 1
        ("Untangling_Digital_Romance.pdf", 0),          # 1. Title Slide
        ("Untangling_Digital_Romance.pdf", 2),          # 2. The Hypothesis & Stakes
        ("Causal_Romance_Engineering.pdf", 2),          # 3. The Illusion of Correlation
        ("Untangling_Digital_Romance.pdf", 1),          # 4. Team Organization
        
        # Speaker 2
        ("Untangling_Digital_Romance.pdf", 3),          # 5. Dataset & Target Variable
        ("Untangling_Digital_Romance.pdf", 4),          # 6. Categorical Uniformity
        ("Untangling_Digital_Romance.pdf", 5),          # 7. Numerical Density & Outliers
        ("Engineering_the_Dating_Pipeline.pdf", 1),     # 8. Protecting Ground Truth (SMOTE Firewall)
        
        # Speaker 3
        ("Engineering_the_Dating_Pipeline.pdf", 0),     # 9. Bulletproofing Compute Infrastructure
        ("Engineering_the_Dating_Pipeline.pdf", 2),     # 10. Pushing 14 Classifiers
        ("Engineering_the_Dating_Pipeline.pdf", 5),     # 11. Brute-Force GPU Optimization
        ("Engineering_the_Dating_Pipeline.pdf", 9),     # 12. Aligning Raw Model Output (Calibration)
        
        # Speaker 4
        ("Engineering_the_Dating_Pipeline.pdf", 3),     # 13. Searching for Non-Linear Synergies (TabNet/GAT)
        ("Engineering_the_Dating_Pipeline.pdf", 4),     # 14. Transductive Pre-Training (SCARF/TabPFN)
        ("SwipeIQ_V2_Strategic_Audit.pdf", 2),          # 15. Algorithmic Recourse: DiCE
        ("SwipeIQ_V2_Strategic_Audit.pdf", 3),          # 16. Causal Uplift Segment Targeting
        
        # Speaker 5
        ("Engineering_the_Dating_Pipeline.pdf", 7),     # 17. Empirical Proof of Absence of Predictive Signal
        ("Engineering_the_Dating_Pipeline.pdf", 6),     # 18. The Ethical Calculus (Pareto)
        ("SwipeIQ_V2_Strategic_Audit.pdf", 1),          # 19. Trustworthy AI: Uncertainty & Fairness
        ("SwipeIQ_V2_Strategic_Audit.pdf", 4),          # 20. SwipeIQ V2 Dashboard
        ("SwipeIQ_V2_Strategic_Evolution.pdf", 4)       # 21. The Future of Algorithmic Romance (Ending Slide)
    ]
    
    writer = PdfWriter()
    
    for filename, page_num in slides_to_extract:
        filepath = os.path.join(base_dir, filename)
        reader = PdfReader(filepath)
        page = reader.pages[page_num]
        writer.add_page(page)
        
    output_path = os.path.join(base_dir, "Absolute_Best_Presentation_Slides_21.pdf")
    with open(output_path, "wb") as f:
        writer.write(f)
        
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    combine_pdfs()
