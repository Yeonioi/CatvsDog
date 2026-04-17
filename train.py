"""
Cat vs Dog Classifier Training Script
Downloads the Microsoft Dogs vs Cats dataset, trains a CNN model,
and saves it for later inference.
"""

import os
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2

# Configuration
IMG_SIZE = 224  # Optimal for 16GB RAM
BATCH_SIZE = 32  # Optimal for 16GB RAM
EPOCHS = 25
MODEL_PATH = 'models/cat_dog_classifier.h5'

def create_model():
    """Create a CNN model with Batch Normalization for better training stability."""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: cat or dog
    ])
    
    return model

def validate_and_clean_images():
    """Validate all images and remove corrupted ones."""
    print("\n🔍 Validating images in data folders...")
    
    from PIL import Image
    removed_count = 0
    
    for split in ['train', 'test']:
        for category in ['cats', 'dogs']:
            folder = f"data/{split}/{category}"
            if not os.path.exists(folder):
                continue
            
            files = os.listdir(folder)
            for file in files:
                filepath = os.path.join(folder, file)
                try:
                    # Try to open and verify image
                    img = Image.open(filepath)
                    img.verify()
                except Exception as e:
                    # Remove corrupted file
                    try:
                        os.remove(filepath)
                        print(f"  ❌ Removed corrupted: {file}")
                        removed_count += 1
                    except:
                        pass
    
    if removed_count > 0:
        print(f"✓ Removed {removed_count} corrupted image(s)")
    else:
        print("✓ All images are valid")

def load_image_data():
    """Load images on-the-fly using ImageDataGenerator for memory efficiency."""
    
    print("\n📂 Preparing data generators...")
    
    # Stronger data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # No augmentation for test data, just rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load from directory - images are loaded in batches, not all at once
    if os.path.exists('data/train'):
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
    else:
        print("⚠ Warning: data/train directory not found")
        return None, None
    
    if os.path.exists('data/test'):
        test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
    else:
        print("⚠ Warning: data/test directory not found")
        test_generator = None
    
    return train_generator, test_generator

def train_model(model):
    """Compile and train the model with memory-efficient data loading."""
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Validate and clean corrupted images first
    validate_and_clean_images()
    
    # Load data generators (images loaded on-the-fly, not all at once!)
    train_generator, test_generator = load_image_data()
    
    if train_generator is None:
        print("\n⚠ No images found. Make sure you have data/train/cats and data/train/dogs directories!")
        return None
    
    print("\nTraining model with memory-efficient batch loading...")
    
    # Define callbacks for better training
    callbacks = [
        # Stop training if validation accuracy doesn't improve
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model automatically
        ModelCheckpoint(
            'models/best_cat_dog_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate if accuracy plateaus
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train with generators - images are loaded one batch at a time
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=callbacks,
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
