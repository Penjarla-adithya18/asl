from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model
logger.info("Loading ASL recognition model...")
try:
    model = load_model('asl_mobilenet_best.h5')  # Using best transfer learning model
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# MediaPipe setup with lower confidence threshold for better detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.3  # Lowered from 0.5 for better detection
)

# ASL alphabet labels (A-Z)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def process_image(image):
    """
    Process image and extract hand region using MediaPipe
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        Processed image ready for model prediction or None if no hand detected
    """
    try:
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hand
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            logger.warning("No hand detected in image")
            return None
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate bounding box with padding
        h, w, c = image.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        padding = 20
        x_min = int(max(0, min(x_coords) - padding))
        x_max = int(min(w, max(x_coords) + padding))
        y_min = int(max(0, min(y_coords) - padding))
        y_max = int(min(h, max(y_coords) + padding))
        
        # Crop hand region
        hand_crop = image[y_min:y_max, x_min:x_max]
        
        if hand_crop.size == 0:
            logger.warning("Hand crop resulted in empty image")
            return None
        
        # Resize to 128x128 (model input size)
        hand_resized = cv2.resize(hand_crop, (128, 128))
        
        # Convert BGR to RGB (MobileNet expects RGB)
        hand_rgb = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        hand_normalized = hand_rgb / 255.0
        
        # Reshape for model input: (batch_size, height, width, channels)
        hand_input = hand_normalized.reshape(1, 128, 128, 3)  # RGB = 3 channels
        
        return hand_input
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model': 'loaded',
        'version': '1.0',
        'labels': len(labels)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Accepts:
        - Multipart form with 'image' file
        - JSON with 'image_base64' field
        
    Returns:
        JSON with letter, confidence, and all predictions
    """
    try:
        # Parse image from request
        image = None
        
        if 'image' in request.files:
            # Handle multipart file upload
            file = request.files['image']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Received image upload: {file.filename}")
            
        elif request.is_json and 'image_base64' in request.json:
            # Handle base64 encoded image
            image_data = base64.b64decode(request.json['image_base64'])
            pil_image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info("Received base64 image")
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process image to extract hand
        processed = process_image(image)
        
        if processed is None:
            return jsonify({
                'error': 'No hand detected',
                'letter': None,
                'confidence': 0
            }), 200  # Return 200 with error message for client to handle
        
        # Make prediction
        prediction = model.predict(processed, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        letter = labels[predicted_class]
        
        logger.info(f"Prediction: {letter} (confidence: {confidence*100:.2f}%)")
        
        # Prepare response with all prediction scores
        all_predictions = {
            labels[i]: round(float(prediction[0][i]) * 100, 2) 
            for i in range(len(labels))
        }
        
        return jsonify({
            'letter': letter,
            'confidence': round(confidence * 100, 2),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        'name': 'ASL Sign Language Recognition API',
        'version': '1.0',
        'model': 'CNN 8 Groups',
        'input_size': '128x128 grayscale',
        'labels': labels,
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict ASL letter from image',
            '/info': 'GET - API information'
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("🚀 Starting ASL Recognition Flask API...")
    logger.info(f"📡 Server will be accessible on port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
