from docx import Document

def main():
    doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
    doc = Document(doc_path)
    
    with open("scratch/docx_paragraphs.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(doc.paragraphs):
            # Only dump paragraphs that have some significant text
            text = p.text.strip()
            if len(text) > 30:
                f.write(f"[{i}] {text}\n")
                
if __name__ == "__main__":
    main()
