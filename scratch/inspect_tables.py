import docx

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report.docx"
doc = docx.Document(doc_path)

for idx, table in enumerate(doc.tables):
    print(f"\n=== Table {idx} (Rows: {len(table.rows)}, Cols: {len(table.columns)}) ===")
    # Print the first row (headers) and first few cells
    for row_idx, row in enumerate(table.rows[:3]):
        row_text = [cell.text.strip() for cell in row.cells]
        print(f"  Row {row_idx}: {row_text}")
    if len(table.rows) > 3:
        print(f"  ... and {len(table.rows) - 3} more rows")
