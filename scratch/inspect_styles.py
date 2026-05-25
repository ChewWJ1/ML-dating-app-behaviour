import docx

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx"
doc = docx.Document(doc_path)

def print_p_details(idx):
    p = doc.paragraphs[idx]
    print(f"\n--- Paragraph {idx} ---")
    print(f"Style Name: {p.style.name}")
    print(f"Paragraph Text: {p.text}")
    print(f"Runs Count: {len(p.runs)}")
    for r_idx, run in enumerate(p.runs):
        print(f"  Run {r_idx}:")
        print(f"    Text: {run.text!r}")
        print(f"    Bold: {run.bold}")
        print(f"    Italic: {run.italic}")
        print(f"    Font Name: {run.font.name}")
        print(f"    Font Size: {run.font.size}")
        print(f"    Font Color: {run.font.color.rgb if run.font.color else None}")

print_p_details(178)
print_p_details(179)
