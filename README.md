# Cat vs Dog Identifier

A beginner-friendly cat vs dog image classifier using TensorFlow/Keras.

## Project Structure

```
CatvsDog/
├── train.py              # Training script
├── inference.py          # Inference/prediction script
├── requirements.txt      # Python dependencies
├── models/               # Directory for saved models
│   └── cat_dog_classifier.h5
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to train the classifier (generates sample data for demonstration):

```bash
python train.py
```

This will:
- Create a CNN model with multiple convolutional layers
- Train on sample data
- Save the model to `models/cat_dog_classifier.h5`

**Note:** For production use, replace the sample data with actual cat and dog images. Download from:
- [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)
- [Kaggle Cats and Dogs](https://www.kaggle.com/datasets/shaunsmith/cat-vs-dog)

### 3. Run Inference

Classify an image:

```bash
python inference.py path/to/your/image.jpg
```

Example output:
```
Prediction: DOG
Confidence: 87.34%
Raw score: 0.8734
```

## Model Architecture

The model uses a Convolutional Neural Network (CNN) with:
- 4 convolutional blocks (each with Conv2D + MaxPooling)
- ReLU activation functions
- Dropout layer (0.5) for regularization
- Sigmoid output for binary classification

```
Input: 224x224 RGB image
↓
Conv2D (32 filters) → MaxPool → Conv2D (64) → MaxPool → 
Conv2D (128) → MaxPool → Conv2D (128) → MaxPool →
Flatten → Dense (512) → Dropout (0.5) → Dense (1, sigmoid)
↓
Output: 0.0 (Cat) or 1.0 (Dog)
```

## Key Parameters

- **Image Size:** 224x224 pixels
- **Batch Size:** 32
- **Epochs:** 10 (increase for better accuracy)
- **Learning Rate:** Adam optimizer (default)
- **Loss Function:** Binary crossentropy

## Next Steps

To improve the model:
1. Use a larger dataset with actual cat/dog images
2. Increase number of epochs during training
3. Data augmentation (rotation, flip, zoom, etc.)
4. Transfer learning with pre-trained models (ResNet, MobileNet)
5. Hyperparameter tuning

## Example: Using Pre-trained Model (Advanced)

For better accuracy out-of-the-box, consider using transfer learning:

```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

## Troubleshooting

**Error: Model not found**
- Run `train.py` first to train and save the model

**Poor predictions**
- Use more training data
- Increase epochs
- Use data augmentation
- Consider transfer learning

**Out of memory**
- Reduce batch size in `train.py` and `inference.py`
- Reduce image size
- Use a simpler model

## License

This project is for educational purposes.
