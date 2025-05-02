import os
import logging
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import processing functions
try:
    from image_processing import run_your_code as run_camera
    from microsoft_tatr import run_your_code as run_scanned
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    raise

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

app = FastAPI()

# Allow requests from localhost during development
origins = [
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "https://my-grid-app.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed images from the output folder
app.mount("/output", StaticFiles(directory="output"), name="output")

class ExtractTextRequest(BaseModel):
    image_url: str

@app.get("/status")
async def status():
    logger.info("Status check successful.")
    return {"status": "Backend is operational."}

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    mode: str = Form(...)  # expects "scanned" or "camera"
):
    logger.info(f"Received file: {file.filename}, Mode: {mode}")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    try:
        # Generate secure temp path
        file_ext = os.path.splitext(file.filename)[1]
        safe_filename = f"{uuid.uuid4()}_{file_ext}"
        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, safe_filename)

        # Save uploaded file
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Process based on mode
        if mode == "scanned":
            result = run_scanned(img_path)
        elif mode == "camera":
            result = run_camera(img_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode selected.")

        # Cleanup temp file
        if os.path.exists(img_path):
            os.remove(img_path)

        # Handle errors in result
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Processing error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        # Normalize camera response
        if mode == "camera":
            result = {
                "images": [result.get("detected_image_url", "/output/default.jpg")],
                "pdf_url": None,
                "cell_coordinates": result.get("cell_coordinates", {}),
            }

        logger.info("Image processed successfully.")
        return JSONResponse(content=result)

    except HTTPException as e:
        logger.warning(f"HTTP Exception: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})