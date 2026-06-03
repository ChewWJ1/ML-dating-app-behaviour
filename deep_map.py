"""
Deep structural map:
- Section 4 (model results) end boundary
- Section 5.2 (explainability) content area
- Section 7.1/7.2 (implemented enhancements) content area
- Section 8.2 (current location of Figs 38-42)
- Exact para indices for each figure block (caption + image para + body + blanks)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from lxml import etree

DOCX_PATH = r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx'
doc = Document(DOCX_PATH)

NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
DRAW_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'

def has_image(para):
    return len(para._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline', )) > 0 or \
           len(para._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}anchor', )) > 0 or \
           'drawing' in etree.tostring(para._element, encoding='unicode').lower()

print("=== FULL STRUCTURAL MAP (para index, style, first 120 chars, has_image) ===")
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    img = '[IMG]' if has_image(p) else ''
    if t or img:
        print(f"  {i:4d} {img:5s} [{p.style.name[:20]:20s}] {t[:110]}")
