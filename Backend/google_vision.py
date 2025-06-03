import requests
import json
import base64
from io import BytesIO
from PIL import Image

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
            raise Exception(f"Error analyzing image: {str(e)}")

    def get_face_analysis(self, image_data):
        """
        Get detailed face analysis from the image
        """
        try:
            result = self.analyze_image(image_data)
            if 'responses' in result and result['responses']:
                face_annotations = result['responses'][0].get('faceAnnotations', [])
                return {
                    'faces_detected': len(face_annotations),
                    'face_details': face_annotations
                }
            return {'faces_detected': 0, 'face_details': []}
        except Exception as e:
            raise Exception(f"Error in face analysis: {str(e)}")

    def get_safe_search(self, image_data):
        """
        Get safe search analysis from the image
        """
        try:
            result = self.analyze_image(image_data)
            if 'responses' in result and result['responses']:
                safe_search = result['responses'][0].get('safeSearchAnnotation', {})
                return safe_search
            return {}
        except Exception as e:
            raise Exception(f"Error in safe search analysis: {str(e)}")

    def get_labels(self, image_data):
        """
        Get image labels and their confidence scores
        """
        try:
            result = self.analyze_image(image_data)
            if 'responses' in result and result['responses']:
                labels = result['responses'][0].get('labelAnnotations', [])
                return [{'description': label['description'], 'confidence': label['score']} 
                       for label in labels]
            return []
        except Exception as e:
            raise Exception(f"Error in label analysis: {str(e)}") 