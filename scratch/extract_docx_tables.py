import sys
import docx

sys.stdout.reconfigure(encoding='utf-8')

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
doc = docx.Document(doc_path)

with open("scratch/docx_tables.txt", "w", encoding="utf-8") as f:
    for i, table in enumerate(doc.tables):
        f.write(f"=== TABLE {i} ===\n")
        for r, row in enumerate(table.rows):
            row_data = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            f.write(f"Row {r}: {' | '.join(row_data)}\n")
        f.write("\n")
