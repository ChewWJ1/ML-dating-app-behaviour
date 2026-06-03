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
        ("Tying_the_Data_Knot.pdf", 0),                 # 1. Title
        ("Engineering_Digital_Romance.pdf", 2),         # 2. Problem Background & Objective
        ("Causal_Romance_Engineering.pdf", 2),          # 3. Illusion of Correlation
        ("Tying_the_Data_Knot.pdf", 1),                 # 4. Pipeline Orchestration / Team
        
        # Speaker 2
        ("Engineering_Digital_Romance.pdf", 3),         # 5. Dataset & Target Variable
        ("Engineering_Digital_Romance.pdf", 4),         # 6. Categorical Uniformity (Fairness)
        ("Causal_Romance_Engineering.pdf", 5),          # 7. Heavy Tails & Zero Linear Signal
        ("Proving_the_Null_Hypothesis.pdf", 1),         # 8. Imbalance Rectification (SMOTE Guardrail)
        
        # Speaker 3
        ("Proving_the_Null_Hypothesis.pdf", 0),         # 9. Hardware Routing
        ("Proving_the_Null_Hypothesis.pdf", 2),         # 10. Baseline Benchmarking (14 Models)
        ("Proving_the_Null_Hypothesis.pdf", 5),         # 11. Massive GPU-Accelerated Optuna Tuning
        ("Proving_the_Null_Hypothesis.pdf", 9),         # 12. Probability Calibration
        
        # Speaker 4
        ("Proving_the_Null_Hypothesis.pdf", 3),         # 13. TabNets & GAT
        ("Proving_the_Null_Hypothesis.pdf", 4),         # 14. Self-Supervised SCARF & Zero-Shot
        ("SwipeIQ_V2_Strategic_Roadmap.pdf", 2),        # 15. Algorithmic Recourse: DiCE
        ("SwipeIQ_V2_Strategic_Roadmap.pdf", 3),        # 16. Causal Uplift Segment Targeting (DML)
        
        # Speaker 5
        ("Proving_the_Null_Hypothesis.pdf", 7),         # 17. The Scientific Truth (ROC 0.50)
        ("Proving_the_Null_Hypothesis.pdf", 6),         # 18. Multi-Objective Pareto Optimization
        ("SwipeIQ_Strategic_ML_Framework.pdf", 1),      # 19. Trustworthy AI (Demographic Parity)
        ("SwipeIQ_V2_Strategic_Roadmap.pdf", 4)         # 20. SwipeIQ V2 Dashboard & Recommendations
    ]
    
    writer = PdfWriter()
    
    for filename, page_num in slides_to_extract:
        filepath = os.path.join(base_dir, filename)
        reader = PdfReader(filepath)
        page = reader.pages[page_num]
        writer.add_page(page)
        
    output_path = os.path.join(base_dir, "Extended_Presentation_Slides_20.pdf")
    with open(output_path, "wb") as f:
        writer.write(f)
        
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    combine_pdfs()
