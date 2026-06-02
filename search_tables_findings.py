import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx')

print("=== SEARCHING TABLES FOR 'finding' OR 'fold' ===")
for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        seen_cells = set()
        for c_idx, cell in enumerate(row.cells):
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            txt = cell.text
            txt_lower = txt.lower()
            if "finding" in txt_lower or "fold" in txt_lower or "cv" in txt_lower:
                print(f"Table {t_idx}, Row {r_idx}, Col {c_idx}:")
                print(txt)
                print("-" * 60)
