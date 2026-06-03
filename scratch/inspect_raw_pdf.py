from pypdf import PdfReader

reader = PdfReader("assets/slides/Proving_the_Null_Hypothesis.pdf")
page = reader.pages[0]
contents = page.get_contents()
print("Contents class:", type(contents))
if contents:
    # If it's a list or similar, let's get the first one or print bytes
    data = contents.get_data() if hasattr(contents, "get_data") else str(contents)
    print("Contents data length:", len(data))
    print("Contents data preview (first 500 bytes):")
    print(data[:500])
