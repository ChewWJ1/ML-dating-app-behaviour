import docx
import os

doc_path = r"reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.1(long).docx"
if not os.path.exists(doc_path):
    print("Error: File not found!")
    exit(1)

doc = docx.Document(doc_path)

def print_p_details(idx):
    p = doc.paragraphs[idx]
    print(f"\n--- Paragraph {idx} ---")
    print(f"Style name: {p.style.name}")
    print(f"Text: {repr(p.text)}")
    print(f"Paragraph alignment: {p.alignment}")
    print(f"Space before: {p.paragraph_format.space_before}")
    print(f"Space after: {p.paragraph_format.space_after}")
    print(f"Line spacing: {p.paragraph_format.line_spacing}")
    print(f"Number of runs: {len(p.runs)}")
    for r_idx, r in enumerate(p.runs):
        print(f"  Run {r_idx}: {repr(r.text)}")
        print(f"    Bold: {r.bold}, Italic: {r.italic}, Underline: {r.underline}")
        if r.font:
            print(f"    Font name: {r.font.name}, Font size: {r.font.size}")

print_p_details(286)
print_p_details(291)
print_p_details(315)
