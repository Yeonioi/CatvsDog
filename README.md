
## Quick Start (No Training)

If you already have a trained model:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place model at `models/best_cat_dog_classifier.h5`

3. Run the web app:
```bash
run_web.bat
```

4. Open browser to **http://localhost:5000**

## Training a New Model

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare training data in `data/train/cats`, `data/train/dogs`, `data/test/cats`, `data/test/dogs`

3. Train the model:
```bash
python train.py
```

4. Run the web app:
```bash
run_web.bat
```

5. Open browser to **http://localhost:5000**

## Usage

- Upload a cat or dog image
- Click "Predict" 
- View classification result with confidence score

---

**Made with Python, TensorFlow, Keras & Flask**
