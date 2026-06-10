import os
import sys

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    from pypdf import PdfReader, PdfWriter

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

import io
from pypdf import PageObject
from pypdf.generic import RectangleObject

def append_png_to_pdf():
    base_dir = r"C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\slides"
    pdf_path = os.path.join(base_dir, "Absolute_Best_Presentation_Slides_21.pdf")
    png_path = r"C:\Users\HP\Downloads\The_Science_of_Connection.png"
    output_path = os.path.join(base_dir, "Absolute_Best_Presentation_Slides_22.pdf")

    # Read the existing PDF
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Copy all existing pages
    for page in reader.pages:
        writer.add_page(page)

    # Get dimensions from the first page to match slide size
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)

    # Open the PNG image
    img = Image.open(png_path)
    
    # Convert to RGB if RGBA
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    # Resize image to match PDF page dimensions while maintaining aspect ratio
    img_ratio = img.width / img.height
    page_ratio = page_width / page_height

    if img_ratio > page_ratio:
        # Image is wider relative to page
        new_width = int(page_width)
        new_height = int(page_width / img_ratio)
    else:
        new_height = int(page_height)
        new_width = int(page_height * img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Save image as a temporary PDF page
    temp_pdf_path = os.path.join(base_dir, "_temp_image_page.pdf")
    
    # Create a new image with exact page dimensions (centered)
    page_img = Image.new('RGB', (int(page_width), int(page_height)), (0, 0, 0))
    x_offset = (int(page_width) - new_width) // 2
    y_offset = (int(page_height) - new_height) // 2
    page_img.paste(img, (x_offset, y_offset))
    
    # Save as PDF
    page_img.save(temp_pdf_path, "PDF", resolution=150.0)
    
    # Read the temp PDF and add to writer
    temp_reader = PdfReader(temp_pdf_path)
    writer.add_page(temp_reader.pages[0])

    # Write the final PDF
    with open(output_path, "wb") as f:
        writer.write(f)
    
    # Clean up temp file
    os.remove(temp_pdf_path)

    print(f"Successfully created {output_path} with {len(writer.pages)} pages")
    print(f"Original had {len(reader.pages)} pages, added 1 PNG as final slide")

if __name__ == "__main__":
    append_png_to_pdf()
