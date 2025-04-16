# image_processing.py
import cv2
import numpy as np
from PIL import Image
import os
import uuid

def run_your_code(image_path):
    # Load and process image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex
    img_output_dir = os.path.join(output_dir, uid)
    os.makedirs(img_output_dir)

    # Crop and save images
    cell_coordinates = []
    cropped_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 700:
            cell_coordinates.append((x, y, w, h))

    sorted_cells = sorted(cell_coordinates, key=lambda r: (r[1], r[0]))
    converted = [(x, y, x + w, y + h) for (x, y, w, h) in sorted_cells]

    for i, (x1, y1, x2, y2) in enumerate(converted):
        cropped = image[y1:y2, x1:x2]
        img_path = os.path.join(img_output_dir, f"cell_{i+1}.png")
        cv2.imwrite(img_path, cropped)
        cropped_images.append(Image.open(img_path))

    if not cropped_images:
        return {"error": "No grid cells detected."}

    # Save to PDF
    pdf_path = os.path.join(img_output_dir, "cropped_output.pdf")
    cropped_images[0].save(pdf_path, save_all=True, append_images=cropped_images[1:])

    # Create public URLs
    img_urls = [
        f"http://localhost:8000/output/{uid}/cell_{i+1}.png" for i in range(len(cropped_images))
    ]
    pdf_url = f"http://localhost:8000/output/{uid}/cropped_output.pdf"

    return {
        "message": "Processing complete!",
        "images": img_urls,
        "pdf_url": pdf_url
    }
