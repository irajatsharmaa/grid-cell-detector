from fastapi import FastAPI, File, UploadFile, HTTPException  # Add File and UploadFile imports
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from image_processing import run_your_code  # Import your image processing function

# Ensure the output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests from React (localhost:3000)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React app to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount the output directory for serving images and PDFs
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
async def root():
    """
    A simple root endpoint to check if the server is running.
    """
    return {"message": "FastAPI Backend is running!"}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to handle image upload, process the image, and return the URLs for cropped images and PDF.
    """
    try:
        # Check if the uploaded file is an image
        if not file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):  # Validate file extension
            raise HTTPException(status_code=400, detail="Only image files are allowed!")

        # Save the uploaded image temporarily
        img_path = f"temp_{file.filename}"
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Run the image processing function (cropping, grid detection, PDF creation)
        result = run_your_code(img_path)

        # Clean up the temporary image after processing
        os.remove(img_path)

        # Check if result contains any errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Return the result with image URLs and PDF download URL
        return JSONResponse(content=result)

    except HTTPException as e:
        # Handle HTTP exceptions (like bad file type)
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        # Handle unexpected errors
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve the files from the output folder
@app.get("/output/{uid}/{filename}")
async def get_file(uid: str, filename: str):
    """
    Endpoint to serve the processed image or PDF files from the output folder.
    """
    file_path = os.path.join("output", uid, filename)
    if os.path.exists(file_path):
        return StaticFiles(directory="output", name=f"{uid}/{filename}")
    else:
        raise HTTPException(status_code=404, detail="File not found.")

# Health check for the server
@app.get("/status")
async def status():
    """
    Endpoint to check the health status of the backend.
    """
    return {"status": "Backend is operational."}
