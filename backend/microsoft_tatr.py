import torch
from PIL import Image, ImageDraw, ImageFont
# from IPython.display import display # Keep if using in Colab/Jupyter
import os # For file path operations
import json # For JSON output
import uuid # For generating unique IDs

# Ensure you have the necessary libraries installed
# !pip install transformers datasets Pillow torch torchvision torchaudio huggingface_hub einops timm

# --- Model Loading Dependencies ---
# Requires: transformers, torch, Pillow, timm, einops
try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    print("Transformers library loaded.")
except ImportError:
    print("Error: 'transformers' library not found.")
    print("Please install it: pip install transformers torch Pillow timm einops")
    exit() # Exit if core dependency is missing

# --- Constants ---
MODEL_CHECKPOINT = "microsoft/table-transformer-structure-recognition"
DEFAULT_PROCESSOR_THRESHOLD = 0.7
DEFAULT_VISUALIZATION_THRESHOLD = 0.8 # Threshold for drawing boxes
OUTPUT_BASE_DIR = "output" # Base directory for all outputs
BASE_URL = "http://localhost:8000" # Base URL for generating links (adjust if needed)


# --- Color Map for Visualization ---
COLOR_MAP = {
    'table': 'red',
    'table row': 'blue',
    'table column': 'green',
    'default': 'orange'
}

# --- Core Detection Function (Unchanged from previous version) ---

def detect_table_elements(imagepath, processor_threshold=DEFAULT_PROCESSOR_THRESHOLD):
    """
    Loads an image, runs the Table Transformer model, and returns the raw
    detections for tables, rows, and columns above a given threshold.

    Args:
        imagepath (str): Path to the input image file.
        processor_threshold (float): Confidence threshold for detection processing.

    Returns:
        tuple: A tuple containing:
            - PIL.Image: The original input image object (or None on error).
            - list: A list of dictionaries for detected tables
                    [{'label': 'table', 'score': float, 'box': [x1,y1,x2,y2]}, ...].
            - list: A list of dictionaries for detected rows
                    [{'label': 'table row', 'score': float, 'box': [x1,y1,x2,y2]}, ...].
            - list: A list of dictionaries for detected columns
                    [{'label': 'table column', 'score': float, 'box': [x1,y1,x2,y2]}, ...].
            Returns (None, [], [], []) if image loading or model processing fails.
    """
    # Load the image
    try:
        image = Image.open(imagepath).convert("RGB")
        original_image_size = image.size
        print(f"Image loaded successfully: {imagepath} ({original_image_size[0]}x{original_image_size[1]})")
    except FileNotFoundError:
        print(f"Error: Image file not found at {imagepath}")
        return None, [], [], []
    except Exception as e:
        print(f"Error opening image '{imagepath}': {e}")
        return None, [], [], []

    # --- Model Loading ---
    print(f"Loading image processor and model from {MODEL_CHECKPOINT}...")
    try:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_CHECKPOINT)
        print("Model and processor loaded.")
    except Exception as e:
        print(f"Error loading model/processor from '{MODEL_CHECKPOINT}': {e}")
        print("Please ensure 'transformers', 'timm', 'einops' are installed,")
        print("you have internet access, or the model is cached correctly.")
        return image, [], [], [] # Return original image, but empty detection lists

    # --- Image Preparation ---
    print("Preparing image for model...")
    try:
        inputs = image_processor(images=image, return_tensors="pt")
        print("Image prepared.")
    except Exception as e:
        print(f"Error processing image with image_processor: {e}")
        return image, [], [], []

    # --- Device Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Using device: {device.upper()}")
    except Exception as e:
        print(f"Error moving model/inputs to device '{device}': {e}")
        if device == 'cuda':
            print("Falling back to CPU.")
            device = 'cpu'
            try:
                model.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e_cpu:
                print(f"Error moving model/inputs to CPU: {e_cpu}")
                return image, [], [], []
        else: # Error even on CPU
             return image, [], [], []

    # --- Model Inference ---
    print("Performing model inference...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print("Inference complete.")
    except Exception as e:
        print(f"Error during model inference: {e}")
        return image, [], [], []

    # --- Post-processing ---
    target_sizes = torch.tensor([original_image_size[::-1]], device=device) # W, H -> H, W
    print(f"Post-processing detections (model output threshold > {processor_threshold})...")
    try:
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=processor_threshold
        )[0]
        results = {k: v.cpu() for k, v in results.items()} # Move results back to CPU
        id2label = model.config.id2label
        print(f"Found {len(results['scores'])} potential detections above threshold {processor_threshold}.")
    except Exception as e:
        print(f"Error during post-processing: {e}")
        return image, [], [], []

    # --- Structure Extraction ---
    all_detections = []
    print("Extracting table, row, and column detections...")
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_label = id2label[label_id.item()]
        if class_label in ['table', 'table row', 'table column']:
            detection_info = {
                'label': class_label,
                'score': round(score.item(), 3),
                'box': [round(coord, 2) for coord in box.tolist()]
            }
            all_detections.append(detection_info)

    tables = sorted([d for d in all_detections if d['label'] == 'table'], key=lambda x: x['box'][1])
    rows = sorted([d for d in all_detections if d['label'] == 'table row'], key=lambda x: x['box'][1])
    columns = sorted([d for d in all_detections if d['label'] == 'table column'], key=lambda x: x['box'][0])

    print(f"Extracted: {len(tables)} tables, {len(rows)} rows, {len(columns)} columns.")

    return image, tables, rows, columns


# --- Visualization Helper Functions (Unchanged from previous version) ---

def draw_detections(image, detections, color_map, font, threshold):
    """Draws bounding boxes for given detections on a copy of the image."""
    if not image:
        print("Error: Cannot draw on None image.")
        return None
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    num_drawn = 0
    for det in detections:
        score = det['score']
        if score >= threshold:
            num_drawn += 1
            box = det['box']
            label = det['label']
            color = color_map.get(label, color_map['default'])
            draw.rectangle(box, outline=color, width=2)
            text = f"{label}: {score:.2f}"
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)
            text_x = box[0]; text_y = box[1] - text_height - 2
            if text_y < 0: text_y = box[1] + 1
            draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=color)
            draw.text((text_x, text_y), text, fill="white", font=font)
    # Only print if there were detections to begin with
    if detections:
        print(f"   Drew {num_drawn} '{detections[0]['label']}' elements (score >= {threshold}).")
    return draw_image

def draw_only_tables(image, tables, color_map, font, threshold):
    """Creates an image showing only detected table boundaries."""
    print(" --> Generating image with Table detections only...")
    return draw_detections(image, tables, color_map, font, threshold)

def draw_rows_and_columns(image, rows, columns, color_map, font, threshold):
    """Creates an image showing detected row and column boundaries."""
    print(" --> Generating image with Row and Column detections...")
    img_with_rows = draw_detections(image, rows, color_map, font, threshold)
    img_with_rows_cols = draw_detections(img_with_rows, columns, color_map, font, threshold)
    return img_with_rows_cols


def run_your_code(image_path):
    try:
        # Font setup
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Output directory
        run_uid = uuid.uuid4().hex
        run_output_dir = os.path.join(OUTPUT_BASE_DIR, run_uid)
        os.makedirs(run_output_dir, exist_ok=True)

        # Output image paths
        output_table_image_path = os.path.join(run_output_dir, "detected_tables_only.png")
        output_row_col_image_path = os.path.join(run_output_dir, "detected_rows_columns.png")

        # Detect elements
        image, tables, rows, columns = detect_table_elements(image_path)

        if not image:
            return {"error": "Failed to load or process image."}

        # Draw and save annotated images
        table_img = draw_only_tables(image, tables, COLOR_MAP, font, DEFAULT_VISUALIZATION_THRESHOLD)
        if table_img:
            table_img.save(output_table_image_path)

        row_col_img = draw_rows_and_columns(image, rows, columns, COLOR_MAP, font, DEFAULT_VISUALIZATION_THRESHOLD)
        if row_col_img:
            row_col_img.save(output_row_col_image_path)

        return {
            "images": [
                f"{BASE_URL}/{OUTPUT_BASE_DIR}/{run_uid}/detected_tables_only.png",
                f"{BASE_URL}/{OUTPUT_BASE_DIR}/{run_uid}/detected_rows_columns.png"
            ],
            "pdf_url": None,
            "tables": tables,
            "rows": rows,
            "columns": columns
        }
    except Exception as e:
        return {"error": str(e)}

# --- Main Execution ---
if __name__ == '__main__':
    # ****** IMPORTANT: REPLACE WITH YOUR IMAGE PATH ******
    input_image_path = '6.jpg' # <--- CHANGE THIS
    # *****************************************************

    # --- Configuration ---
    proc_threshold = DEFAULT_PROCESSOR_THRESHOLD # Model's internal filtering threshold
    vis_threshold = DEFAULT_VISUALIZATION_THRESHOLD # Threshold for drawing boxes

    # --- Font setup ---
    try:
        font = ImageFont.truetype("arial.ttf", 12) # Try to load Arial
    except IOError:
        print("Arial font not found, using default PIL font.")
        font = ImageFont.load_default() # Fallback to default font

    # --- Check if image exists ---
    if not os.path.exists(input_image_path):
        print("-" * 50)
        print(f"ERROR: Input image path '{input_image_path}' does not exist.")
        print("Please replace the placeholder path in the script with a valid path.")
        print("-" * 50)
    else:
        # --- Prepare Output Directory ---
        run_uid = uuid.uuid4().hex # Generate unique ID for this run
        run_output_dir = os.path.join(OUTPUT_BASE_DIR, run_uid)
        try:
            os.makedirs(run_output_dir, exist_ok=True) # Create base 'output' dir and unique subdir
            print(f"Created output directory: {run_output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{run_output_dir}': {e}")
            # Decide if you want to exit or try saving in the current dir as fallback
            exit() # Exit if directory creation fails

        # --- Define Output File Paths ---
        output_table_image_path = os.path.join(run_output_dir, "detected_tables_only.png")
        output_row_col_image_path = os.path.join(run_output_dir, "detected_rows_columns.png")
        output_json_path = os.path.join(run_output_dir, "detected_structures.json")

        # --- Run Detection ---
        original_image, tables, rows, columns = detect_table_elements(
            imagepath=input_image_path,
            processor_threshold=proc_threshold
        )

        # --- Process Results ---
        if original_image and (tables or rows or columns):
            print("\n--- Processing and Saving Results ---")

            # 1. Generate and save image with only tables
            table_image = draw_only_tables(original_image, tables, COLOR_MAP, font, vis_threshold)
            if table_image:
                try:
                    table_image.save(output_table_image_path)
                    print(f" --> Table detection image saved to: '{output_table_image_path}'")
                except Exception as e:
                    print(f"Error saving table image: {e}")


            # 2. Generate and save image with rows and columns
            row_col_image = draw_rows_and_columns(original_image, rows, columns, COLOR_MAP, font, vis_threshold)
            if row_col_image:
                try:
                    row_col_image.save(output_row_col_image_path)
                    print(f" --> Row/Column detection image saved to: '{output_row_col_image_path}'")
                except Exception as e:
                    print(f"Error saving row/column image: {e}")

            # 3. Prepare JSON data
            output_data = {
                "source_image_path": input_image_path,
                "output_directory_id": run_uid,
                "processing_threshold": proc_threshold,
                "tables": tables,
                "rows": rows,
                "columns": columns
            }

            # 4. Save JSON data
            try:
                with open(output_json_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                print(f" --> Detection data saved as JSON to: '{output_json_path}'")
            except Exception as e:
                print(f"Error saving JSON data: {e}")

            # --- Generate and Print URLs (Example) ---
            print("\n--- Generated URLs (Example) ---")
            # Construct URLs relative to the base output directory and UID
            table_image_url = f"{BASE_URL}/{OUTPUT_BASE_DIR}/{run_uid}/detected_tables_only.png"
            row_col_image_url = f"{BASE_URL}/{OUTPUT_BASE_DIR}/{run_uid}/detected_rows_columns.png"
            json_url = f"{BASE_URL}/{OUTPUT_BASE_DIR}/{run_uid}/detected_structures.json"

            print(f" Table Image URL: {table_image_url}")
            print(f" Row/Col Image URL: {row_col_image_url}")
            print(f" JSON Data URL: {json_url}")


            # --- Optional: Display Summary ---
            # (Removed detailed summary print here for brevity, but can be added back)
            print(f"\n--- Detection Summary ---")
            print(f" Tables Detected (above proc_threshold): {len(tables)}")
            print(f" Rows Detected (above proc_threshold): {len(rows)}")
            print(f" Columns Detected (above proc_threshold): {len(columns)}")


        elif original_image:
             print(f"\nNo table structures (tables, rows, or columns) detected above the processing threshold {proc_threshold}.")
        else:
            print("\nImage processing failed. Cannot generate outputs.")