import pypdf
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.2 (final).pdf"
if not os.path.exists(pdf_path):
    print(f"Error: PDF not found at {pdf_path}!")
    sys.exit(1)

print(f"Loading PDF from {pdf_path}...")
reader = pypdf.PdfReader(pdf_path)
print(f"Total Pages in PDF: {len(reader.pages)}")

# Extract all text to do global search
full_text = ""
pages_text = []
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    pages_text.append(text)
    full_text += f"\n--- PAGE {idx + 1} ---\n" + text

print("\n==========================================")
print("AUDITING STRUCTURE & HEADINGS")
print("==========================================")

headings = [
    "5.0 Insights and Interpretation",
    "6.0 SwipeIQ V2: Premium Interactive Analytics Dashboard and Web Application",
    "6.1 Engineering Architecture and Cloud Deployment Framework",
    "6.2 Detailed Specifications of Interactive Workspaces & Stress-Testing Playgrounds",
    "6.3 System Engineering & Modular Implementation Specifications",
    "7.0 Implemented Enhancements, Performance Optimization & Excluded Techniques",
    "7.1 Summary of Implemented Enhancements & Optimizations",
    "7.2 Detailed Technical Specifications",
    "7.3 Summary of Evaluated and Excluded Techniques",
    "8.0 Conclusion and Future Work",
    "8.1 Key Findings Summary",
    "8.2 Recommendations for Future Research"
]

missing_headings = []
for h in headings:
    if h.lower() in full_text.lower():
        # Find which page it is on
        found_page = -1
        for p_idx, p_txt in enumerate(pages_text):
            if h.lower() in p_txt.lower():
                found_page = p_idx + 1
                break
        print(f"  [OK] Found heading: '{h}' on Page {found_page}")
    else:
        print(f"  [MISSING] Heading: '{h}'")
        missing_headings.append(h)

print("\n==========================================")
print("AUDITING METRICS & TABLES")
print("==========================================")

# Search for the Table 6 and 7 metrics in the text
metrics_to_check = {
    "KNN Baseline F1 (54.00%)": "54.00%",
    "LightGBM Accuracy (58.62%)": "58.62%",
    "KNN Recall (84.16%)": "84.16%",
    "SVM Baseline Accuracy (60.30%)": "60.30%",
    "CatBoost in Table 8": "CatBoost"
}

for desc, val in metrics_to_check.items():
    if val in full_text:
        found_pages = []
        for p_idx, p_txt in enumerate(pages_text):
            if val in p_txt:
                found_pages.append(str(p_idx + 1))
        print(f"  [OK] Found metric '{desc}' ({val}) on page(s): {', '.join(found_pages)}")
    else:
        print(f"  [MISSING] Metric: '{desc}' ({val})")

# Look at Table 6 / 7 / 8 text segments
print("\n--- Examining Text around Table 6 / 7 / 8 ---")
for p_idx, p_txt in enumerate(pages_text):
    if "Table 6" in p_txt:
        print(f"\n[Page {p_idx + 1} - Section around Table 6]:")
        # Print a snippet of page text around it
        lines = p_txt.split("\n")
        t6_lines = [l for l in lines if any(x in l for x in ["Table 6", "K-Nearest Neighbors", "LightGBM", "54.00%", "58.62%"])]
        for l in t6_lines[:10]:
            print(f"  {l}")

    if "Table 8" in p_txt:
        print(f"\n[Page {p_idx + 1} - Section around Table 8]:")
        lines = p_txt.split("\n")
        t8_lines = [l for l in lines if any(x in l for x in ["Table 8", "CatBoost", "LightGBM", "31.46%", "35.14%"])]
        for l in t8_lines[:10]:
            print(f"  {l}")

print("\n==========================================")
print("AUDITING LINK & DUST / CHARMAP CHECKS")
print("==========================================")

# Search for Streamlit hyperlink
app_url = "https://ml-tying-the-data-knot-swipeiq-app.streamlit.app/"
if app_url in full_text:
    found_page = -1
    for p_idx, p_txt in enumerate(pages_text):
        if app_url in p_txt:
            found_page = p_idx + 1
            break
    print(f"  [OK] Live App Link found on Page {found_page}")
else:
    print("  [ERROR] Live App Link is missing from the PDF text!")

# Check for the replacement characters / encoding glitches (e.g.  or visual artifacts)
bad_chars = ["", "", ""]
found_bad = False
for bad in bad_chars:
    if bad in full_text:
        found_bad = True
        found_pages = []
        for p_idx, p_txt in enumerate(pages_text):
            if bad in p_txt:
                found_pages.append(str(p_idx + 1))
        print(f"  [WARNING] Found potential encoding glitch character '{bad}' on page(s): {', '.join(found_pages)}")
if not found_bad:
    print("  [OK] Zero encoding replacement characters () found in the PDF text!")

# Print Figure sequence around Figure 10 / Figure 37
print("\n--- Sequential Figure check in final sections ---")
lines = full_text.split("\n")
for idx, line in enumerate(lines):
    if "Figure " in line:
        # Check if it has a figure label
        if any(f"Figure {n}:" in line or f"Figure {n} " in line for n in range(10, 44)):
            print(f"  Line: {line[:120]}")
