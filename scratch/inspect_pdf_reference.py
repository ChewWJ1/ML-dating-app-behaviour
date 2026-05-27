import pypdf

pdf_path = "reports/WIA1006 Machine Learning - Tying the Data Knot Assignment Report.pdf"
reader = pypdf.PdfReader(pdf_path)

print(f"Total Pages in Reference PDF: {len(reader.pages)}")

# Let's inspect the text for headings
headings = []
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    for line in text.split('\n'):
        line_strip = line.strip()
        # Headings are usually numbered like 1.0, 1.1, etc.
        if len(line_strip) > 3 and line_strip[0].isdigit() and ('.' in line_strip[:5] or line_strip.startswith("Table") or line_strip.startswith("Figure")):
            headings.append((idx + 1, line_strip))

print("\n--- Outlines/References Found in PDF ---")
for page_num, h in headings[:100]:
    print(f"Page {page_num}: {h[:120]}")
