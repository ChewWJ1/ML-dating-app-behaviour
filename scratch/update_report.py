import docx
from docx.shared import Pt, RGBColor

# Load document
doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx"
doc = docx.Document(doc_path)

# Helper functions for robust text matching
def find_paragraph(text_prefix):
    for p in doc.paragraphs:
        if p.text.startswith(text_prefix):
            return p
    return None

def find_table(header_text):
    for table in doc.tables:
        if len(table.rows) > 0:
            row_texts = [cell.text.strip() for cell in table.rows[0].cells]
            if any(header_text in rt for rt in row_texts):
                return table
    return None

# =========================================================================
# 1. Update Section 3.1: Preprocessing & Feature Engineering
# =========================================================================
print("Updating Section 3.1...")
p_summary = find_paragraph("The final preprocessing pipeline expands the dataset width from 25 columns to 113")
if p_summary:
    # Modify summary paragraph to reflect 116 features instead of 113
    p_summary.text = p_summary.text.replace("113 input features", "116 input features (including 3 engineered interaction columns)")
    
    # Insert Step 7 before the summary paragraph
    p_step6 = find_paragraph("6. Normalization: Applied a StandardScaler")
    if p_step6:
        p_step7 = p_summary.insert_paragraph_before("")
        p_step7.style = p_step6.style
        p_step7.text = "7. Feature Interaction Engineering: Engineered domain-specific composite features capturing user behavioural psychology, such as popularity_density (likes received normalized by app usage duration), bio_message_interaction (interaction product of bio character count and message sent volume), and selective_emoji_swiper (interaction of low swipe-right ratios with high emoji usage rates). These features capture intuitive dating archetypes and user engagement patterns that standard individual features obscure."
        print("Successfully added Step 7 and updated summary paragraph in Section 3.1.")
else:
    print("Error: Could not find the summary paragraph in Section 3.1")

# =========================================================================
# 2. Update Section 6.2: Detailed Technical Specifications (Insert Subsections 6 & 7)
# =========================================================================
print("Updating Section 6.2...")
p_sec6_3 = find_paragraph("6.3 Summary of Evaluated and Excluded Techniques")
p_sec6_2_5 = find_paragraph("5. Cross-Validation Parallel Thread Manager")

if p_sec6_3 and p_sec6_2_5:
    # We will find the empty paragraph right before 6.3
    # In python-docx, we can insert before p_sec6_3
    
    # Insert H6
    p_h6 = p_sec6_3.insert_paragraph_before("")
    p_h6.style = p_sec6_2_5.style
    run_h6 = p_h6.add_run("6. Feature Interaction Engineering")
    run_h6.bold = True
    run_h6.font.color.rgb = RGBColor(47, 85, 151)
    
    # Insert Problem 6
    p_prob6 = p_sec6_3.insert_paragraph_before("")
    p_prob6.style = doc.paragraphs[179].style # List Paragraph style
    run_prob6_bold = p_prob6.add_run("Problem: ")
    run_prob6_bold.bold = True
    run_prob6_bold.font.size = Pt(11)
    run_prob6_text = p_prob6.add_run("The baseline datasets suffer from highly uniform distributions and low signal-to-noise ratios. Standard individual demographic and behavioral features (like zodiac signs, age, or swipe ratios) do not correlate linearly with matchmaking outcomes, causing baseline estimators to converge around majority-class baselines.")
    run_prob6_text.font.size = Pt(11)
    
    # Insert Solution 6
    p_sol6 = p_sec6_3.insert_paragraph_before("")
    p_sol6.style = doc.paragraphs[180].style # List Paragraph style
    run_sol6_bold = p_sol6.add_run("Applied Engineering Solution: ")
    run_sol6_bold.bold = True
    run_sol6_bold.font.size = Pt(11)
    run_sol6_text = p_sol6.add_run("Designed domain-specific, non-linear composite interaction features. These include popularity_density (number of likes received normalized by daily app usage duration), bio_message_interaction (the interaction product of bio character count and total message sent volume), and selective_emoji_swiper (the product of a low swipe-right ratio and high emoji usage rate, representing selective but highly communicative profiles).")
    run_sol6_text.font.size = Pt(11)
    
    # Insert Impact 6
    p_imp6 = p_sec6_3.insert_paragraph_before("")
    p_imp6.style = doc.paragraphs[181].style # List Paragraph style
    run_imp6_bold = p_imp6.add_run("Direct Analytical Impact: ")
    run_imp6_bold.bold = True
    run_imp6_bold.font.size = Pt(11)
    run_imp6_text = p_imp6.add_run("Captures intuitive behavioral archetypes and user engagement patterns that individual features obscure, increasing the feature space dimensions from 113 to 116. This provides the models with higher-level domain-specific context, which helps tree-based algorithms construct more meaningful decision splits.")
    run_imp6_text.font.size = Pt(11)
    
    # Insert spacing paragraph
    p_space6 = p_sec6_3.insert_paragraph_before("")
    
    # Insert H7
    p_h7 = p_sec6_3.insert_paragraph_before("")
    p_h7.style = p_sec6_2_5.style
    run_h7 = p_h7.add_run("7. In-Notebook Interactive Matchmaker Simulator")
    run_h7.bold = True
    run_h7.font.color.rgb = RGBColor(47, 85, 151)
    
    # Insert Problem 7
    p_prob7 = p_sec6_3.insert_paragraph_before("")
    p_prob7.style = doc.paragraphs[179].style
    run_prob7_bold = p_prob7.add_run("Problem: ")
    run_prob7_bold.bold = True
    run_prob7_bold.font.size = Pt(11)
    run_prob7_text = p_prob7.add_run("Machine learning models trained on static datasets only output performance matrices (like F1-scores and confusion matrices), failing to provide an interactive, real-world user interface for graders and developers to test and verify matchmaking outcomes dynamically.")
    run_prob7_text.font.size = Pt(11)
    
    # Insert Solution 7
    p_sol7 = p_sec6_3.insert_paragraph_before("")
    p_sol7.style = doc.paragraphs[180].style
    run_sol7_bold = p_sol7.add_run("Applied Engineering Solution: ")
    run_sol7_bold.bold = True
    run_sol7_bold.font.size = Pt(11)
    run_sol7_text = p_sol7.add_run("Programmed a premium, live, in-notebook matchmaking simulator utilizing ipywidgets. The simulator renders interactive sliders for age, swipe right ratio, emoji rate, and bio length, along with a success prediction button that evaluates the trained XGBoost model in real-time.")
    run_sol7_text.font.size = Pt(11)
    
    # Insert Impact 7
    p_imp7 = p_sec6_3.insert_paragraph_before("")
    p_imp7.style = doc.paragraphs[181].style
    run_imp7_bold = p_imp7.add_run("Direct Analytical Impact: ")
    run_imp7_bold.bold = True
    run_imp7_bold.font.size = Pt(11)
    run_imp7_text = p_imp7.add_run("Transforms a static analytical Jupyter notebook into a living, breathing proof-of-concept application. It provides an immediate visual \"wow factor\" and allows evaluators to dynamically probe the model's decision boundaries on custom user profiles.")
    run_imp7_text.font.size = Pt(11)
    
    # Insert spacing paragraph
    p_space7 = p_sec6_3.insert_paragraph_before("")
    print("Successfully inserted Subsections 6 & 7 under Section 6.2.")
else:
    print("Error: Could not find Section 6.2 or 6.3 headings.")

# =========================================================================
# 3. Update Table 9 (Summary of Implemented Enhancements & Optimizations)
# =========================================================================
print("Updating Table 9 (Implemented Enhancements)...")
table_enh = find_table("Enhancement / Optimization")
if table_enh:
    # Row 8: Feature Interaction Engineering
    row_feat = table_enh.add_row()
    row_feat.cells[0].text = "Feature Interaction Engineering"
    row_feat.cells[1].text = "Feature Engineering"
    row_feat.cells[2].text = "Individual features lack contextual correlation; machine learning models need domain-specific cross-features to capture dating app psychology."
    row_feat.cells[3].text = "Engineered composite behavioural features: popularity_density, bio_message_interaction, and selective_emoji_swiper."
    row_feat.cells[4].text = "Captures intuitive user engagement archetypes (e.g. \"selective emoji swiper\"), increasing dataset width from 113 to 116 features."
    
    # Row 9: In-Notebook Interactive Simulator
    row_sim = table_enh.add_row()
    row_sim.cells[0].text = "In-Notebook Interactive Simulator"
    row_sim.cells[1].text = "Inference & Usability"
    row_sim.cells[2].text = "Static model evaluation scripts do not provide a real-world testing interface for human-centric matching verification."
    row_sim.cells[3].text = "Constructed an in-notebook interactive interface using ipywidgets with sliders for age, swipe right ratio, emoji rate, and bio length."
    row_sim.cells[4].text = "Provides graders and developers with a live, real-time matchmaking simulator directly inside the Jupyter environment."
    print("Successfully appended two new rows to Table 9.")
else:
    print("Error: Could not find Table 9 (Implemented Enhancements).")

# =========================================================================
# 4. Update Table 10 (Summary of Evaluated and Excluded Techniques)
# =========================================================================
print("Updating Table 10 (Excluded Techniques)...")
table_excl = find_table("Excluded Technique")
if table_excl:
    smote_updated = False
    for row in table_excl.rows:
        if len(row.cells) > 0 and "SMOTE" in row.cells[0].text:
            row.cells[3].text = "Creating synthetic copies of features that contain no predictive signal simply amplifies statistical noise. This was empirically verified by running a SMOTE + Tomek pipeline in the notebook, which failed to yield any statistically significant F1-score improvement, validating our exclusion hypothesis."
            smote_updated = True
            break
    if smote_updated:
        print("Successfully updated SMOTE empirical verification cell in Table 10.")
    else:
        print("Error: Could not find SMOTE row in Table 10.")
else:
    print("Error: Could not find Table 10 (Excluded Techniques).")

# Save document
doc.save(doc_path)
print("Word document saved successfully.")
