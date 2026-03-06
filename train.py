"""
Cat vs Dog Classifier Training Script
Downloads the Microsoft Dogs vs Cats dataset, trains a CNN model,
and saves it for later inference.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'models/cat_dog_classifier.h5'

def download_and_prepare_data():
    """Download the cats and dogs dataset from Microsoft."""
    print("Downloading dataset...")
    
    # Download the dataset
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    path_to_zip = "cats_and_dogs.zip"
    
    # Alternative: Use TensorFlow datasets
    # This is simpler for beginners
    try:
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        print("Note: Using CIFAR-10 as a fallback. For actual cats/dogs, download from:")
        print("https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765")
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    return None

def create_model():
    """Create a CNN model for cat vs dog classification."""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: cat or dog
    ])
    
    return model

def load_image_data():
    """Load images from the data directory."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import cv2
    
    print("\n📂 Loading images from data directory...")
    
    train_images = []
    train_labels = []
    
    # Load training images
    for label, category in enumerate(["cats", "dogs"]):
        folder = f"data/train/{category}"
        if not os.path.exists(folder):
            print(f"⚠ Warning: {folder} not found")
            continue
        
        files = os.listdir(folder)
        for i, file in enumerate(files):
            try:
                img = cv2.imread(os.path.join(folder, file))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    train_images.append(img)
                    train_labels.append(label)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1}/{len(files)} {category}")
            except Exception as e:
                pass
    
    # Load test images
    test_images = []
    test_labels = []
    
    for label, category in enumerate(["cats", "dogs"]):
        folder = f"data/test/{category}"
        if not os.path.exists(folder):
            continue
        
        files = os.listdir(folder)
        for i, file in enumerate(files):
            try:
                img = cv2.imread(os.path.join(folder, file))
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    test_images.append(img)
                    test_labels.append(label)
                
                if (i + 1) % 500 == 0:
                    print(f"  Loaded {i + 1}/{len(files)} test {category}")
            except Exception as e:
                pass
    
    X_train = np.array(train_images, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.float32)
    X_test = np.array(test_images, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.float32)
    
    print(f"\n✓ Loaded training: {X_train.shape}")
    print(f"✓ Loaded test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def train_model(model):
    """Compile and train the model."""
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Load real cat/dog images
    X_train, y_train, X_test, y_test = load_image_data()
    
    if X_train.size == 0:
        print("\n⚠ No images found. Make sure you ran download_dataset.py first!")
        return None
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history

def save_model(model):
    """Save the trained model."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def main():
    print("=" * 50)
    print("Cat vs Dog Classifier - Training")
    print("=" * 50)
    
    # Create model
    model = create_model()
    
    # Train model
    history = train_model(model)
    
    # Save model
    save_model(model)
    
    print("\nTraining complete!")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
