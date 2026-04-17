"""
Cat vs Dog Classifier - Flask Web App
"""

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow import keras
import os
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model configuration
IMG_SIZE = 224
MODEL_PATH = 'models/best_cat_dog_classifier.h5'
UNCERTAIN_MARGIN = 0.15
MIN_CONFIDENCE = 0.70  # Reject predictions below 70% confidence

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run train.py first to train the model.")
else:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

def preprocess_image(image):
    """Preprocess image for prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array
    image_array = np.array(image).astype(np.float32)
    
    # Normalize
    image_array = image_array / 255.0
    
    # Ensure image shape is always (H, W, 3)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
        raise ValueError("Unsupported image format. Please use a standard JPG or PNG image.")
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    if image_array.shape != (1, IMG_SIZE, IMG_SIZE, 3):
        raise ValueError("Image preprocessing failed due to invalid tensor shape.")
    
    return image_array

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load image
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        dog_prob = float(prediction)
        cat_prob = 1.0 - dog_prob
        top_prob = max(cat_prob, dog_prob)
        
        # Interpret result
        if top_prob < MIN_CONFIDENCE:
            label = "Low Confidence (Uncertain)"
            confidence = top_prob * 100
            color = "#d97706"
            message = "Confidence is too low. This might not be a clear cat or dog photo. Try another image."
        elif abs(prediction - 0.5) <= UNCERTAIN_MARGIN:
            label = "Neither Cat nor Dog (Uncertain)"
            confidence = top_prob * 100
            color = "#d97706"
            message = "The model is not confident this image is a clear cat or dog. Try another photo."
        elif prediction < 0.5:
            label = "CAT"
            confidence = cat_prob * 100
            color = "#e74c3c"
            message = "Looks like a cat based on model confidence."
        else:
            label = "DOG"
            confidence = dog_prob * 100
            color = "#3498db"
            message = "Looks like a dog based on model confidence."
        
        return jsonify({
            'label': label,
            'confidence': round(confidence, 1),
            'color': color,
            'message': message,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Could not classify this image. Details: {str(e)}',
            'success': False
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
