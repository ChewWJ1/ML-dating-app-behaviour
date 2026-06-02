import docx

def main():
    doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
    doc = docx.Document(doc_path)
    
    with open("scratch/docx_full_text.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(doc.paragraphs):
            text = p.text.strip()
            if text:
                f.write(f"[{i}] {text}\n")

if __name__ == "__main__":
    main()
