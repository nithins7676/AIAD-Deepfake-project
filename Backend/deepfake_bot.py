import os
import io
from io import BytesIO
import base64
import random
import torch
import timm
import numpy as np
import cv2
import requests
import json
from PIL import Image
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY', None)  # Optional for reverse image search

class GoogleVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.vision_api_url = 'https://vision.googleapis.com/v1/images:annotate'

    def analyze_image(self, image_data):
        """
        Analyze image using Google Cloud Vision API
        """
        try:
            # Convert image to base64
            if isinstance(image_data, bytes):
                image_content = base64.b64encode(image_data).decode('utf-8')
            else:
                # If image_data is a file path or PIL Image
                if isinstance(image_data, str):
                    with open(image_data, 'rb') as image_file:
                        image_content = base64.b64encode(image_file.read()).decode('utf-8')
                elif isinstance(image_data, Image.Image):
                    buffered = BytesIO()
                    image_data.save(buffered, format="JPEG")
                    image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    raise ValueError("Unsupported image data type")

            # Prepare the request
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_content
                        },
                        "features": [
                            {
                                "type": "FACE_DETECTION",
                                "maxResults": 10
                            },
                            {
                                "type": "LABEL_DETECTION",
                                "maxResults": 10
                            },
                            {
                                "type": "SAFE_SEARCH_DETECTION"
                            },
                            {
                                "type": "WEB_DETECTION",
                                "maxResults": 20  # Increase max results for better source detection
                            }
                        ]
                    }
                ]
            }

            # Make the API request
            response = requests.post(
                f"{self.vision_api_url}?key={self.api_key}",
                json=request_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return None

    def get_face_analysis(self, image_data):
        """
        Get detailed face analysis from the image
        """
        try:
            result = self.analyze_image(image_data)
            if result and 'responses' in result and result['responses']:
                face_annotations = result['responses'][0].get('faceAnnotations', [])
                return {
                    'faces_detected': len(face_annotations),
                    'face_details': face_annotations
                }
            return {'faces_detected': 0, 'face_details': []}
        except Exception as e:
            print(f"Error in face analysis: {str(e)}")
            return {'faces_detected': 0, 'face_details': []}

    def get_safe_search(self, image_data):
        """
        Get safe search analysis from the image
        """
        try:
            result = self.analyze_image(image_data)
            if result and 'responses' in result and result['responses']:
                safe_search = result['responses'][0].get('safeSearchAnnotation', {})
                return safe_search
            return {}
        except Exception as e:
            print(f"Error in safe search analysis: {str(e)}")
            return {}

    def get_labels(self, image_data):
        """
        Get image labels and their confidence scores
        """
        try:
            result = self.analyze_image(image_data)
            if result and 'responses' in result and result['responses']:
                labels = result['responses'][0].get('labelAnnotations', [])
                return [{'description': label['description'], 'confidence': label['score']} 
                        for label in labels]
            return []
        except Exception as e:
            print(f"Error in label analysis: {str(e)}")
            return []
            
    def get_web_detection(self, image_data):
        """
        Get web entities and matching images with detailed source information
        """
        try:
            result = self.analyze_image(image_data)
            if result and 'responses' in result and result['responses']:
                web_detection = result['responses'][0].get('webDetection', {})
                return web_detection
            return {}
        except Exception as e:
            print(f"Error in web detection: {str(e)}")
            return {}

class DeepfakeDetector:
    def __init__(self, model_path="best_vit_model.pth"):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = timm.create_model("mvitv2_base_cls", pretrained=False, num_classes=2)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            print("Attempting to load from fallback location...")
            try:
                self.model.load_state_dict(torch.load("mvitv2.pkl", map_location=self.device))
                print("Model loaded from mvitv2.pkl")
            except Exception as e2:
                print(f"Error loading model from fallback location: {str(e2)}")
                raise Exception("Failed to load model from all locations")
                
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def detect_faces(self, img_cv):
        """Detect faces in an OpenCV image"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Try different scale factors for better face detection
        scale_factors = [1.1, 1.05, 1.2]
        min_neighbors_options = [5, 3, 7]
        
        faces = []
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_options:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break
                
        return faces, gray
    
    def predict(self, image, process_full_image=False):
        """
        Predict if the image is a deepfake
        
        Args:
            image: PIL Image
            process_full_image: Whether to process the full image if no face is detected
            
        Returns:
            dict: Prediction results
        """
        # Convert PIL image to CV2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces, gray = self.detect_faces(img_cv)
        
        if len(faces) == 0:
            if process_full_image:
                print("No face detected. Processing the entire image.")
                face_pil = image
            else:
                return {
                    "error": "No face detected in the image.",
                    "face_detected": False
                }
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
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()

        return {
            "prediction": "Real" if predicted_class == 1 else "Manipulated",
            "confidence": {
                "real": float(probs[1].item()),
                "manipulated": float(probs[0].item())
            },
            "face_detected": len(faces) > 0,
            "multiple_faces": len(faces) > 1
        }

class ReverseImageSearch:
    """Advanced reverse image search with multiple methods"""
    
    def __init__(self, serp_api_key=None, vision_api=None):
        self.serp_api_key = serp_api_key
        self.vision_api = vision_api
        
    async def search_with_vision_api(self, image):
        """Search using Google Vision API web detection"""
        if not self.vision_api:
            return None
            
        try:
            web_detection = self.vision_api.get_web_detection(image)
            
            # Process results
            exact_matches = []
            page_matches = []
            similar_matches = []
            
            # Process full matching images (highest confidence)
            if 'fullMatchingImages' in web_detection:
                for match in web_detection['fullMatchingImages']:
                    url = match.get('url', '')
                    if url:
                        exact_matches.append({
                            'url': url,
                            'domain': self._extract_domain(url),
                            'confidence': 0.9,
                            'match_type': 'exact'
                        })
            
            # Process pages with matching images
            if 'pagesWithMatchingImages' in web_detection:
                for page in web_detection['pagesWithMatchingImages']:
                    url = page.get('url', '')
                    if url:
                        page_matches.append({
                            'url': url,
                            'domain': self._extract_domain(url),
                            'confidence': 0.8,
                            'match_type': 'page'
                        })
            
            # Process partial matching images
            if 'partialMatchingImages' in web_detection:
                for match in web_detection['partialMatchingImages']:
                    url = match.get('url', '')
                    if url:
                        similar_matches.append({
                            'url': url,
                            'domain': self._extract_domain(url),
                            'confidence': 0.6,
                            'match_type': 'partial'
                        })
            
            # Process visually similar images
            if 'visuallySimilarImages' in web_detection:
                for match in web_detection['visuallySimilarImages']:
                    url = match.get('url', '')
                    if url:
                        similar_matches.append({
                            'url': url,
                            'domain': self._extract_domain(url),
                            'confidence': 0.4,
                            'match_type': 'similar'
                        })
                        
            # Extract web entities
            web_entities = []
            if 'webEntities' in web_detection:
                for entity in web_detection['webEntities']:
                    if 'description' in entity and entity.get('score', 0) > 0.5:
                        web_entities.append({
                            'name': entity['description'],
                            'score': entity.get('score', 0)
                        })
            
            # Return structured results
            results = {
                'exact_matches': exact_matches,
                'page_matches': page_matches,
                'similar_matches': similar_matches,
                'web_entities': web_entities
            }
            
            return results
                
        except Exception as e:
            print(f"Error in Vision API image search: {str(e)}")
            return None
            
    async def search_with_serp_api(self, image):
        """Search using SerpAPI for Google Lens results"""
        if not self.serp_api_key:
            return None
            
        try:
            # Convert image to temporary base64 for Google reverse image search
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            url = f"https://serpapi.com/search.json?engine=google_lens&api_key={self.serp_api_key}"
            payload = {"image": img_str}
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Extract source domains
                source_dict = {}
                
                # Extract from visual matches
                if "visual_matches" in data:
                    for match in data["visual_matches"]:
                        if "link" in match:
                            url = match["link"]
                            domain = self._extract_domain(url)
                            if domain and domain not in source_dict:
                                source_dict[domain] = {
                                    'url': url,
                                    'confidence': match.get('confidence', 0.7),
                                    'match_type': 'visual',
                                    'title': match.get('title', '')
                                }
                                
                # Extract from knowledge graph if available
                if "knowledge_graph" in data and "source" in data["knowledge_graph"]:
                    source = data["knowledge_graph"]["source"]
                    domain = self._extract_domain(source)
                    if domain and domain not in source_dict:
                        source_dict[domain] = {
                            'url': source,
                            'confidence': 0.9,
                            'match_type': 'knowledge_graph',
                            'title': data["knowledge_graph"].get('title', '')
                        }
                        
                return {
                    'source_domains': [
                        {
                            'domain': domain,
                            'url': data['url'],
                            'confidence': data['confidence'],
                            'match_type': data['match_type'],
                            'title': data.get('title', '')
                        } for domain, data in source_dict.items()
                    ]
                }
                
            return None
                
        except Exception as e:
            print(f"Error in SerpAPI image search: {str(e)}")
            return None
    
    def _extract_domain(self, url):
        """Extract domain from a URL"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove 'www.' prefix for consistency
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain if domain else None
        except:
            return None
            
    async def search(self, image):
        """
        Comprehensive reverse image search with multiple methods
        Returns exact image URLs where the image appears
        """
        results = {
            'exact_matches': [],
            'page_matches': [],
            'similar_matches': [],
            'web_entities': []
        }
        
        # Try Google Vision API first (preferred method)
        vision_results = await self.search_with_vision_api(image)
        if vision_results:
            if 'exact_matches' in vision_results and vision_results['exact_matches']:
                results['exact_matches'].extend(vision_results['exact_matches'])
            
            if 'page_matches' in vision_results and vision_results['page_matches']:
                results['page_matches'].extend(vision_results['page_matches'])
                
            if 'similar_matches' in vision_results and vision_results['similar_matches']:
                results['similar_matches'].extend(vision_results['similar_matches'])
                
            if 'web_entities' in vision_results and vision_results['web_entities']:
                results['web_entities'] = vision_results['web_entities']
        
        # Ensure no duplicate URLs
        seen_urls = set()
        for key in ['exact_matches', 'page_matches', 'similar_matches']:
            filtered = []
            for item in results[key]:
                if item['url'] not in seen_urls:
                    filtered.append(item)
                    seen_urls.add(item['url'])
            results[key] = filtered
        
        return results

# Initialize APIs and detector
try:
    vision_api = GoogleVisionAPI(GOOGLE_VISION_API_KEY) if GOOGLE_VISION_API_KEY else None
    deepfake_detector = DeepfakeDetector()
    reverse_search = ReverseImageSearch(serp_api_key=SERP_API_KEY, vision_api=vision_api)
except Exception as e:
    print(f"Error initializing APIs: {str(e)}")
    print("The bot will start, but some features may be limited.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Deepfake Detection Bot!\n\n"
        "Send me any image and I'll analyze it for:\n"
        "• Deepfake detection\n"
        "• Original image sources\n"
        "• Face and content analysis\n\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Available Commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Simply send any image to analyze it for deepfakes and get detailed insights including where the image originally appeared online!"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle images sent by users"""
    try:
        # Send processing message
        processing_message = await update.message.reply_text("🔄 Analyzing image... Please wait.")
        
        # Get the photo file
        photo = await update.message.photo[-1].get_file()
        
        # Download the photo
        photo_bytes = await photo.download_as_bytearray()
        
        # Create a PIL Image from bytes
        image = Image.open(BytesIO(photo_bytes))
        
        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n🔄 Running deepfake detection...")
        
        # Get deepfake prediction
        deepfake_result = deepfake_detector.predict(image, process_full_image=True)
        
        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n✅ Deepfake detection complete\n🔄 Analyzing image content...")
        
        # Get Google Vision analysis
        face_analysis = {'faces_detected': 0}
        safe_search = {}
        labels = []
        
        if vision_api:
            try:
                face_analysis = vision_api.get_face_analysis(image)
                safe_search = vision_api.get_safe_search(image)
                labels = vision_api.get_labels(image)
            except Exception as e:
                print(f"Google Vision API error: {str(e)}")

        # Update processing message
        await processing_message.edit_text("🔍 Analyzing image...\n✅ Deepfake detection complete\n✅ Content analysis complete\n🔄 Searching for exact image matches online...")
        
        # Get image sources
        search_result = await reverse_search.search(image)
        
        # Format the response
        if "error" in deepfake_result:
            response = f"⚠️ {deepfake_result['error']}\n\nTry sending a clearer image with visible faces."
        else:
            response = (
                f"🔍 Image Analysis Results\n\n"
                f"🤖 Deepfake Detection:\n"
                f"• Status: {deepfake_result['prediction']}\n"
                f"• Confidence:\n"
                f"  - Real: {deepfake_result['confidence']['real']*100:.2f}%\n"
                f"  - Manipulated: {deepfake_result['confidence']['manipulated']*100:.2f}%\n"
            )
            
            # Add face information
            if deepfake_result['face_detected']:
                response += f"\n👤 Face Analysis:\n"
                response += f"• Faces Detected: {face_analysis['faces_detected']}\n"
                response += f"• Multiple Faces: {'Yes' if deepfake_result.get('multiple_faces', False) else 'No'}\n"
            
            # Add top labels
            if labels:
                response += f"\n🏷️ Image Content:\n"
                for label in labels[:3]:
                    response += f"• {label['description']} ({label['confidence']*100:.1f}%)\n"
            
            # Add exact matching image URLs
            has_results = False
            
            if search_result and search_result['exact_matches']:
                has_results = True
                response += f"\n🔍 Exact Image Matches:\n"
                for i, match in enumerate(search_result['exact_matches'][:3], 1):
                    domain = match.get('domain', reverse_search._extract_domain(match['url']))
                    response += f"{i}. <a href=\"{match['url']}\">{domain}</a>\n"
            
            if search_result and search_result['page_matches']:
                has_results = True
                response += f"\n📄 Pages with this Image:\n"
                for i, match in enumerate(search_result['page_matches'][:3], 1):
                    domain = match.get('domain', reverse_search._extract_domain(match['url']))
                    response += f"{i}. <a href=\"{match['url']}\">{domain}</a>\n"
                    
            if search_result and search_result['similar_matches']:
                has_results = True
                response += f"\n🔍 Similar Images:\n"
                for i, match in enumerate(search_result['similar_matches'][:3], 1):
                    domain = match.get('domain', reverse_search._extract_domain(match['url']))
                    response += f"{i}. <a href=\"{match['url']}\">{domain}</a>\n"
                    
            # Add web entities if available
            if search_result and 'web_entities' in search_result and search_result['web_entities']:
                response += f"\n🌐 Web Entities:\n"
                for entity in search_result['web_entities'][:3]:
                    response += f"• {entity['name']} ({entity['score']*100:.1f}%)\n"
                    
            if not has_results:
                response += f"\n🔎 No matching images found online. This may be a new or private image."
        
        # Send the results
        await processing_message.edit_text(response, parse_mode='HTML')
        
    except Exception as e:
        error_message = f"❌ Error analyzing image: {str(e)}"
        if 'processing_message' in locals():
            await processing_message.edit_text(error_message)
        else:
            await update.message.reply_text(error_message)

def main():
    # Check if Telegram token is available
    if not TELEGRAM_TOKEN:
        print("❌ Error: TELEGRAM_TOKEN not found in environment variables or .env file.")
        return
        
    # Check if the model was initialized successfully
    if 'deepfake_detector' not in globals():
        print("❌ Error: DeepfakeDetector could not be initialized.")
        return
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Start the bot
    print("🤖 Starting Deepfake Detection Telegram Bot...")
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for inference")
    print(f"Google Vision API: {'Enabled' if vision_api else 'Disabled'}")
    print(f"SerpAPI: {'Enabled' if SERP_API_KEY else 'Disabled'}")
    application.run_polling()

if __name__ == "__main__":
    main() 