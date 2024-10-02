import pymupdf
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import json

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# Create directories for storing output images and JSON files
output_dir = "output"
images_dir = os.path.join(output_dir, "images")
json_dir = os.path.join(output_dir, "json")

if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

# Open the PDF file using PyMuPDF (pymupdf)
pdf = pymupdf.open("research_paper.pdf")

# Set Pixels Per Inch (PPI) for image rendering
ppi = 100

# Loop through each page in the PDF
for page_num in range(pdf.page_count):
    page = pdf.load_page(page_num)  # Load the page

    # Render the page as an image
    pix = page.get_pixmap(dpi=ppi)
    image_path = os.path.join(images_dir, f"page_{page_num + 1}.png")
    pix.save(image_path)

    # Perform OCR on the image
    ocr_result = ocr.ocr(image_path, cls=True)

    # Analyze the OCR results to detect structure
    page_content = {"page_num": page_num + 1, "text": [], "structure": []}

    for res in ocr_result:
        for line in res:
            # Extract bounding box, text, and confidence score
            box = line[0]
            text = line[1][0]
            score = line[1][1]

            # Add text and structure information to JSON
            page_content["text"].append({"text": text, "confidence": score, "box": box})

            # Structure detection: basic check for paragraphs or tables (for simplicity)
            if "table" in text.lower():
                page_content["structure"].append("table")
            elif len(text.split()) > 5:  # Simple heuristic to identify paragraphs
                page_content["structure"].append("paragraph")
            else:
                page_content["structure"].append("other")

    # Save OCR results in JSON format for each page
    json_output_path = os.path.join(json_dir, f"page_{page_num + 1}.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(page_content, json_file, indent=4)

    # Save OCR-annotated image with boxes (Optional for visualization)
    image = Image.open(image_path).convert('RGB')
    boxes = [elements[0] for elements in ocr_result[0]]
    txts = [elements[1][0] for elements in ocr_result[0]]
    scores = [elements[1][1] for elements in ocr_result[0]]

    im_show = draw_ocr(image, boxes, txts, scores, font_path='Ubuntu-L.ttf')
    im_show = Image.fromarray(im_show)
    annotated_image_path = os.path.join(images_dir, f"annotated_page_{page_num + 1}.png")
    im_show.save(annotated_image_path)

    print(f"Page {page_num + 1} processed and saved as {image_path} with annotation {annotated_image_path}")

print("OCR and structure detection completed. Results saved in separate folders for images and JSON.")
