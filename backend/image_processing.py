# image_processing.py
import cv2
import numpy as np
# from PIL import Image # PIL is no longer needed for PDF generation
import os
import uuid

def run_your_code(image_path):
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not load image at path: {image_path}"}

    # Create a copy to draw on, keeping the original clean
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    # Increase iterations slightly if needed for clearer lines
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

    # Find and sort cell coordinates
    cell_coordinates_xywh = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Keep a reasonable area threshold to filter noise
        if w * h > 700:
            cell_coordinates_xywh.append((x, y, w, h))

    # Check if any cells were detected before proceeding
    if not cell_coordinates_xywh:
        return {"error": "No grid cells detected."}

    # Sort cells primarily by row (y-coordinate), then by column (x-coordinate)
    sorted_cells = sorted(cell_coordinates_xywh, key=lambda r: (r[1], r[0]))

    # Draw rectangles for detected cells on the output image
    for i, (x, y, w, h) in enumerate(sorted_cells):
        # Draw a green rectangle with thickness 2
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Optional: Add text label (cell number) near the rectangle
        # cv2.putText(output_image, str(i+1), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # --- Removed individual cell cropping and saving logic ---
    # --- Removed PDF generation logic ---

    # Save the single image with detected cells marked
    detected_image_filename = f"detected_cells_{uid}.png"
    detected_image_path = os.path.join(img_output_dir, detected_image_filename)
    save_success = cv2.imwrite(detected_image_path, output_image)

    if not save_success:
         return {"error": f"Failed to save the detected image to {detected_image_path}"}

    # Create public URL for the single detected image
    # Ensure your server is configured to serve files from the 'output' directory
    # Adjust base URL if needed (e.g., if running behind a proxy or different host/port)
    base_url = "http://localhost:8000" # Or your actual server base URL
    detected_image_url = f"{base_url}/output/{uid}/{detected_image_filename}"

    # Return the URL of the single image and the list of coordinates
    return {
        "message": f"Processing complete! {len(sorted_cells)} cells detected.",
        "detected_image_url": detected_image_url,
        "cell_coordinates": sorted_cells  # List of (x, y, w, h) tuples
    }

# Example Usage (if run directly, replace with actual image path)
if __name__ == '__main__':
    # Create a dummy image for testing if needed
    # test_image_path = 'path/to/your/grid_image.png'
    # if not os.path.exists(test_image_path):
    #    print(f"Test image not found at {test_image_path}. Please provide a valid path.")
    # else:
    #    result = run_your_code(test_image_path)
    #    print(result)
    print("This script should be called with an image path.")
    # Example call:
    # result = run_your_code('my_table.png')
    # print(result)