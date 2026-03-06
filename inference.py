"""
Cat vs Dog Classifier Inference Script
Uses the trained model to classify images as cats or dogs.
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'models/cat_dog_classifier.h5'

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model.")
        sys.exit(1)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Normalize pixel values (0-1)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict(model, image_path):
    """Predict whether an image contains a cat or dog."""
    # Preprocess image
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0][0]
    
    # Interpret prediction
    # prediction close to 0 = cat
    # prediction close to 1 = dog
    confidence = max(prediction, 1 - prediction) * 100
    
    if prediction < 0.5:
        label = "CAT"
        score = (1 - prediction) * 100
    else:
        label = "DOG"
        score = prediction * 100
    
    return {
        'label': label,
        'confidence': score,
        'raw_prediction': prediction
    }

def main():
    print("=" * 50)
    print("Cat vs Dog Classifier - Inference")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    # For demonstration, use a sample image
    print("\nUsage Examples:")
    print("  python inference.py path/to/image.jpg")
    print("  python inference.py cat.png")
    print("  python inference.py /path/to/dog.jpg")
    
    # If image path provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        print(f"\nAnalyzing: {image_path}")
        result = predict(model, image_path)
        
        if result:
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Raw score: {result['raw_prediction']:.4f}")
    else:
        print("\nNo image path provided.")
        print("Please provide an image path as argument:")
        print("  python inference.py <path_to_image>")

if __name__ == "__main__":
    main()
