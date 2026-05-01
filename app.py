"""
Cat vs Dog vs Other Classifier — Flask Web App (3-class)
"""

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow import keras
import os
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Config (must match train.py) ─────────────────────────────────────────────
IMG_SIZE       = 128
MODEL_PATH     = os.path.join('models', 'best_model.keras')
CLASS_NAMES    = ['cat', 'dog', 'other']     # must match folder order in data/train
CLASS_COLORS   = {
    'cat':   '#e74c3c',
    'dog':   '#3498db',
    'other': '#6b7280',
}
CLASS_LABELS   = {'cat': 'Cat', 'dog': 'Dog', 'other': 'Other'}

MIN_CONFIDENCE    = 0.70   # below this → "low confidence"
UNCERTAIN_MARGIN  = 0.15   # top-2 gap below this → "uncertain"
OTHER_THRESHOLD   = 0.35   # at or below this → default to "other" (no confidence bar)
TTA_RUNS          = 6      # 1 clean + 5 augmented (batched in one predict call)

# ── Load model ───────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"Model not found at '{MODEL_PATH}'. Run train.py first.")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_single(image: Image.Image, augment: bool = False) -> np.ndarray:
    """Return a (IMG_SIZE, IMG_SIZE, 3) float32 array normalised to [0, 1]."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if augment:
        # Subtle augmentations matching training config
        angle = np.random.uniform(-15, 15)
        image = image.rotate(angle, expand=False)

        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        brightness = np.random.uniform(0.85, 1.15)
        image = ImageEnhance.Brightness(image).enhance(brightness)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0

    # Ensure exactly 3 channels
    if arr.ndim == 2:                          # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:                   # RGBA → RGB
        arr = arr[:, :, :3]

    if arr.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Unexpected image shape after preprocessing: {arr.shape}")

    return arr


def predict_with_tta(image: Image.Image, n_runs: int = TTA_RUNS) -> np.ndarray:
    """
    Run TTA: batch all augmented versions in ONE model.predict() call.
    Returns mean probability vector of shape (NUM_CLASSES,).
    """
    frames = [preprocess_single(image, augment=False)]           # clean pass first
    for _ in range(n_runs - 1):
        frames.append(preprocess_single(image, augment=True))

    batch = np.stack(frames, axis=0)                             # (n_runs, H, W, 3)
    preds = model.predict(batch, verbose=0)                      # (n_runs, NUM_CLASSES)
    return preds.mean(axis=0)                                    # (NUM_CLASSES,)


# ── Uncertainty helpers ───────────────────────────────────────────────────────
def softmax_entropy(probs: np.ndarray) -> float:
    """Normalised entropy in [0, 1]. 1 = maximally uncertain."""
    eps = 1e-7
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy)


def top2_gap(probs: np.ndarray) -> float:
    """Difference between the top-2 class probabilities."""
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[0] - sorted_p[1])


def make_other_response():
    """Return the standard payload for 'not a cat or dog' results."""
    return dict(
        label='Not a Cat or Dog',
        color=CLASS_COLORS['other'],
        confidence=None,
        message="This picture is not a cat or dog. Please try a different picture.",
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Run train.py first.', 'success': False}), 503

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.', 'success': False}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.', 'success': False}), 400

        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception:
            return jsonify({'error': 'Could not open image. Upload a valid JPG or PNG.', 'success': False}), 400

        try:
            mean_probs = predict_with_tta(image)                     # (NUM_CLASSES,)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500

        top_idx   = int(np.argmax(mean_probs))
        top_prob  = float(mean_probs[top_idx])
        top_label = CLASS_NAMES[top_idx]
        entropy   = softmax_entropy(mean_probs)
        gap       = top2_gap(mean_probs)

        # ── Decision logic ────────────────────────────────────────────────────
        # Hard floor: 30% or below → always "other", no confidence bar
        if top_prob <= 0.30:
            result = make_other_response()

        # Near-uniform distribution across all 3 classes
        elif entropy > 0.75:
            if (1 - entropy) <= OTHER_THRESHOLD:
                # Entropy so high the model is essentially guessing → other
                result = make_other_response()
            else:
                result = dict(
                    label='Uncertain',
                    color='#d97706',
                    confidence=round((1 - entropy) * 100, 1),
                    message=(
                        "The model is highly uncertain. This image may not contain "
                        "a recognisable cat or dog."
                    ),
                )

        # Dominant class found but not confident enough
        elif top_prob < MIN_CONFIDENCE:
            if top_prob <= OTHER_THRESHOLD:
                # Below 35% → treat as other, no confidence bar
                result = make_other_response()
            else:
                result = dict(
                    label='Low Confidence',
                    color='#d97706',
                    confidence=round(top_prob * 100, 1),
                    message=(
                        f"Best guess is {top_label.upper()} ({top_prob * 100:.1f}%) "
                        f"but confidence is too low. Try a clearer photo."
                    ),
                )

        # Two classes are very close — model is on the fence
        elif gap < UNCERTAIN_MARGIN:
            if top_prob <= OTHER_THRESHOLD:
                # Even the top score is too low → other, no confidence bar
                result = make_other_response()
            else:
                second_idx   = int(np.argsort(mean_probs)[-2])
                second_label = CLASS_NAMES[second_idx]
                result = dict(
                    label='Uncertain',
                    color='#d97706',
                    confidence=round(top_prob * 100, 1),
                    message=(
                        f"Model is torn between {top_label.upper()} and "
                        f"{second_label.upper()} (gap: {gap * 100:.1f}%). "
                        f"Try another photo."
                    ),
                )

        # Clear prediction
        else:
            if top_label == 'other':
                # Explicit "other" class prediction → no confidence bar
                result = make_other_response()
            else:
                result = dict(
                    label=top_label.upper(),
                    color=CLASS_COLORS[top_label],
                    confidence=round(top_prob * 100, 1),
                    message=f"This looks like a {top_label} with {round(top_prob * 100, 1):.1f}% confidence.",
                )

        # All class probabilities for optional display in the UI
        all_probs = {
            CLASS_NAMES[i]: round(float(mean_probs[i]) * 100, 1)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            'success':    True,
            'label':      result['label'],
            'confidence': result['confidence'],
            'color':      result['color'],
            'message':    result['message'],
            'all_probs':  all_probs,
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}', 'success': False}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)