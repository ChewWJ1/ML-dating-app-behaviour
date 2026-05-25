import docx
import sys

# Set standard output to UTF-8 to prevent cp1252 crash
sys.stdout.reconfigure(encoding='utf-8')

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx"
doc = docx.Document(doc_path)

for idx, table in enumerate(doc.tables):
    print(f"\n=== Table {idx} (Rows: {len(table.rows)}, Cols: {len(table.columns)}) ===")
    for row_idx, row in enumerate(table.rows):
        row_text = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        print(f"  Row {row_idx}: {row_text[:4]} ... ({len(row_text)} cells total)")
