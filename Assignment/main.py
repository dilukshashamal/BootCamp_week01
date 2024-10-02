import pymupdf
from paddleocr import PaddleOCR, PPStructure, draw_ocr, draw_structure_result, save_structure_res
from PIL import Image
import os
import json
import cv2

# Initialize PaddleOCR and PPStructure for OCR and layout detection
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
layout_model = PPStructure(layout=True, lang='en')

# Create directories for storing output images, JSON, and layout
output_dir = "output"
images_dir = os.path.join(output_dir, "images")
json_dir = os.path.join(output_dir, "json")
layout_dir = os.path.join(output_dir, "layout")

# Ensure the directories exist
# loops through each of the previously defined directories, check if they exit
for dir_path in [images_dir, json_dir, layout_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Open the PDF file
pdf = pymupdf.open("research_paper.pdf")

# Set Pixels Per Inch for image rendering
ppi = 100

for page_num in range(pdf.page_count):
    page = pdf.load_page(page_num)

    # Converts the PDF page into an image
    pix = page.get_pixmap(dpi=ppi)
    image_path = os.path.join(images_dir, f"page_{page_num + 1}.png")
    pix.save(image_path)

    # Runs OCCR on the image
    ocr_result = ocr.ocr(image_path, cls=True)

    # Prepare for Storing OCR and Structure Data
    page_content = {"page_num": page_num + 1, "text": [], "structure": []}

    # Iterate thogh the OCR results for each line, extracting the bounding box, recognized text and confidence score
    for res in ocr_result:
        for line in res:
            # Extract bounding box, text, and confidence score
            box = line[0]
            text = line[1][0]
            score = line[1][1]

            # Add text and structure information to JSON
            page_content["text"].append({"text": text, "confidence": score, "box": box})

            # Basic check for paragraphs or tables
            # if "table" in text.lower():
            #     page_content["structure"].append("table")
            # elif len(text.split()) > 5:  
            #     page_content["structure"].append("paragraph")
            # else:
            #     page_content["structure"].append("other")

    # Save OCR results in JSON format for each page
    json_output_path = os.path.join(json_dir, f"page_{page_num + 1}.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(page_content, json_file, indent=4)

    # Perform layout detection using PPStructure
    image_cv2 = cv2.imread(image_path)
    layout_result = layout_model(image_cv2)

    # Save layout results
    save_structure_res(layout_result, layout_dir, f"layout_page_{page_num + 1}")

    # Visualize layout detection results and save
    image_pil = Image.open(image_path).convert('RGB')
    im_layout = draw_structure_result(image_pil, layout_result, font_path='Ubuntu-L.ttf')
    layout_image_path = os.path.join(layout_dir, f"layout_page_{page_num + 1}.png")
    im_layout_pil = Image.fromarray(im_layout)
    im_layout_pil.save(layout_image_path)

    # Save OCR-annotated image with bounding boxes
    boxes = [elements[0] for elements in ocr_result[0]]
    txts = [elements[1][0] for elements in ocr_result[0]]
    scores = [elements[1][1] for elements in ocr_result[0]]
    im_show = draw_ocr(image_pil, boxes, txts, scores, font_path='Ubuntu-L.ttf')
    im_show = Image.fromarray(im_show)
    annotated_image_path = os.path.join(images_dir, f"annotated_page_{page_num + 1}.png")
    im_show.save(annotated_image_path)

    print(f"Page {page_num + 1} processed: OCR image saved at {image_path}, layout saved at {layout_image_path}")

print("OCR and layout detection completed. Results saved in separate folders for images, layouts, and JSON.")
