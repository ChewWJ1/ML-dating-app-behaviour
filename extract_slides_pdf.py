from PyPDF2 import PdfReader
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\slides\Absolute_Best_Presentation_Slides_21.pdf'
reader = PdfReader(path)
print(f"Total pages: {len(reader.pages)}\n")

for i, page in enumerate(reader.pages, 1):
    text = page.extract_text()
    print(f"=== PAGE {i} ===")
    if text and text.strip():
        print(text)
    else:
        print("[IMAGE-ONLY PAGE - No extractable text]")
    print()
