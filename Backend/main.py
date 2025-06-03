from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import io
import random
from google_vision import GoogleVisionAPI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')

# Initialize Google Vision API
vision_api = GoogleVisionAPI(GOOGLE_VISION_API_KEY)

# 🔒 Make model deterministic
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 🧠 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("mvitv2_base_cls", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
model.to(device)
model.eval()

# 🎨 Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🧍 Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 🚀 FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), process_full_image: bool = False):
    try:
        print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
        
        # Check file size before reading
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Read file in chunks to check size
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                return JSONResponse(
                    {"error": f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB."},
                    status_code=413
                )
        
        # Reset file position for actual processing
        await file.seek(0)
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        if len(contents) == 0:
            return JSONResponse(
                {"error": "Empty file received"},
                status_code=400
            )
            
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"Image size: {image.size}")
        except Exception as e:
            return JSONResponse(
                {"error": f"Invalid or unsupported image format. Error: {str(e)}"},
                status_code=400
            )

        # Get Google Vision API analysis
        try:
            vision_analysis = vision_api.analyze_image(image)
            face_analysis = vision_api.get_face_analysis(image)
            safe_search = vision_api.get_safe_search(image)
            labels = vision_api.get_labels(image)
        except Exception as e:
            print(f"Google Vision API error: {str(e)}")
            vision_analysis = None
            face_analysis = None
            safe_search = None
            labels = None

        # Continue with existing face detection and deepfake analysis
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Try different scale factors for better face detection
        scale_factors = [1.1, 1.05, 1.2]
        min_neighbors_options = [5, 3, 7]
        
        faces = []
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_options:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break
                
        if len(faces) == 0:
            if process_full_image:
                print("No face detected. Processing the entire image as requested.")
                face_pil = image
            else:
                return JSONResponse(
                    {"error": "No face detected in the image. Try setting process_full_image=true to analyze the entire image."},
                    status_code=400
                )
        else:
            if len(faces) > 1:
                print(f"Multiple faces detected ({len(faces)}). Using the largest one.")
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
            else:
                x, y, w, h = faces[0]
                
            face = img_cv[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Get deepfake prediction
        input_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()

        result = {
            "prediction": "Real" if predicted_class == 1 else "Manipulated",
            "confidence": {
                "real": float(probs[1].item()),
                "manipulated": float(probs[0].item())
            },
            "face_detected": len(faces) > 0,
            "multiple_faces": len(faces) > 1,
            "google_vision": {
                "face_analysis": face_analysis,
                "safe_search": safe_search,
                "labels": labels
            } if vision_analysis else None
        }
        print(f"Prediction result: {result}")
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
