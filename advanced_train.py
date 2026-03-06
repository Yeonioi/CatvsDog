"""
Advanced Cat vs Dog Classifier using Transfer Learning
Uses a pre-trained MobileNetV2 model for better accuracy and faster training.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = 'models/cat_dog_classifier_advanced.h5'

def create_transfer_learning_model():
    """Create a model using transfer learning with MobileNetV2."""
    print("Loading pre-trained MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model
    model = models.Sequential([
        base_model,
        
        # Global average pooling to convert spatial dimensions to a single vector
        layers.GlobalAveragePooling2D(),
        
        # Dense layers for classification
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

def create_data_augmentation():
    """Create data augmentation for training."""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def train_model(model, base_model):
    """Compile and train the model."""
    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # For demonstration, create random data
    # In production, load actual images from directories
    print("\nGenerating sample data...")
    X_train = np.random.rand(100, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    
    X_test = np.random.rand(20, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, 20)
    
    print("Training model...")
    history = model.fit(
        X_train / 255.0, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test / 255.0, y_test),
        verbose=1
    )
    
    # Optional: Fine-tune base model
    print("\nFine-tuning base model...")
    base_model.trainable = True
    
    # Only fine-tune top layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Fine-tuning...")
    history_ft = model.fit(
        X_train / 255.0, y_train,
        batch_size=BATCH_SIZE,
        epochs=2,
        validation_data=(X_test / 255.0, y_test),
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
    print("Advanced Cat vs Dog - Transfer Learning")
    print("=" * 50)
    
    # Create model with transfer learning
    model, base_model = create_transfer_learning_model()
    
    # Train model
    history = train_model(model, base_model)
    
    # Save model
    save_model(model)
    
    print("\nAdvanced training complete!")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
