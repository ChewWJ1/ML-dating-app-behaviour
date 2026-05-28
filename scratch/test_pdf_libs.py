import sys

libs = ["pypdf", "PyPDF2", "pdfplumber", "fitz", "pdfminer"]
available = []

for lib in libs:
    try:
        __import__(lib)
        available.append(lib)
    except ImportError:
        pass

print(f"Python version: {sys.version}")
print(f"Available PDF libraries: {available}")
