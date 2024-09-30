
import fitz  # PyMuPDF
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Open the PDF file
pdf = fitz.open("research_paper.pdf")

# 1. How many pages are in the PDF document?
print("Number of pages in the PDF document:", pdf.page_count)

# 2. What is the height of the first page?
first_page = pdf[0]
print("Height of the first page:", first_page.mediabox[3])

# 3. What is the width of the first page?
print("Width of the first page:", first_page.mediabox[2])

# Select the first page (index 0)
first_page = pdf[0]

# Get the page size in points
page_size = first_page.mediabox_size

# Convert points to inches (1 point = 1/72 inch)
width_in_inches = page_size[0] / 72
height_in_inches = page_size[1] / 72

# Convert inches to pixels at 100 PPI
ppi = 100
width_in_pixels = width_in_inches * ppi
height_in_pixels = height_in_inches * ppi

print(f"Width: {width_in_pixels} pixels, Height: {height_in_pixels} pixels at {ppi} PPI")

# Load the first page and convert it to an image
first_page = pdf.load_page(0)
pix = first_page.get_pixmap(dpi=ppi)
output_path = "first_page_image.png"
pix.save(output_path)

# Loop through each page in the PDF and save as PNG images
for page_num in range(pdf.page_count):
    # Load the page
    page = pdf.load_page(page_num)

    # Render the page as an image
    pix = page.get_pixmap(dpi=ppi)

    # Define the output image path
    output_image_path = f"png/page_{page_num + 1}.png"

    # Save the image
    pix.save(output_image_path)

    print(f"Page {page_num + 1} saved as PNG at {output_image_path}")

# Extract text from the first page and save to a text file
first_page_text = first_page.get_text().encode("utf8")
with open("first_page_text.txt", "wb") as out:
    out.write(first_page_text)

# OCR processing
ocr = PaddleOCR(use_angle_cls=True, lang="en")
image_path = "first_page_image.png"

ocr_result = ocr.ocr(image_path, cls=True)

# Print OCR results
for idx in range(len(ocr_result)):
    res = ocr_result[idx]
    for line in res:
        print(line)

# Draw OCR results on the image
result = ocr_result[0]
image = Image.open(image_path).convert('RGB')

boxes = [elements[0] for elements in result]
txts = [elements[1][0] for elements in result]
scores = [elements[1][1] for elements in result]

im_show = draw_ocr(image, boxes, txts, scores, font_path='Ubuntu-L.ttf')
im_show = Image.fromarray(im_show)
im_show.save("first_page_annotation.png")
