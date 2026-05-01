"""
Cat vs Dog vs Other Classifier — Optimized Training Script
3-class: cats, dogs, other
Training from scratch on CPU (i7-8700K)
"""

import os
import numpy as np

# ── CPU thread config (i7-8700K: 6 cores / 12 threads) ──────────────────────
os.environ['TF_NUM_INTRAOP_THREADS'] = '10'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'   # suppress TF info spam

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = 128        # Faster per-epoch on CPU; bump to 160 if accuracy plateaus
BATCH_SIZE  = 32
EPOCHS      = 60
NUM_CLASSES = 3
L2          = 1e-4
MODEL_DIR   = 'models'
BEST_MODEL  = os.path.join(MODEL_DIR, 'best_model.keras')
FINAL_MODEL = os.path.join(MODEL_DIR, 'final_model.keras')
DATA_TRAIN  = 'data/train'
DATA_VAL    = 'data/val'    # keep a proper val split separate from test
DATA_TEST   = 'data/test'


# ── Image validation ─────────────────────────────────────────────────────────
def validate_and_clean_images():
    """Remove corrupted images from ALL split/class folders."""
    from PIL import Image
    print("\nValidating images...")
    removed = 0
    for split in ['train', 'val', 'test']:
        split_path = f'data/{split}'
        if not os.path.exists(split_path):
            continue
        for cls in os.listdir(split_path):
            folder = os.path.join(split_path, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                try:
                    img = Image.open(fpath)
                    img.verify()
                except Exception:
                    try:
                        os.remove(fpath)
                        print(f"  Removed corrupted: {fpath}")
                        removed += 1
                    except Exception:
                        pass
    print(f"Validation done — {removed} corrupted file(s) removed.\n")


# ── Model ────────────────────────────────────────────────────────────────────
def residual_block(x, filters, stride=1):
    """
    Pre-activation residual block (ResNet v2 style).
    BN → ReLU → Conv → BN → ReLU → Conv + skip
    """
    shortcut = x

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)

    # Projection shortcut when dimensions change
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False,
                                 kernel_regularizer=regularizers.l2(L2))(shortcut)

    return layers.Add()([x, shortcut])


def create_model():
    """
    Compact ResNet-v2 built for scratch training on CPU.
    Depth is kept reasonable so each epoch completes in ~10-15 min on i7-8700K.
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Stem
    x = layers.Conv2D(32, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1 — 32 filters
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Stage 2 — 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.15)(x)

    # Stage 3 — 128 filters
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # Stage 4 — 256 filters (wider instead of deeper to save time)
    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # Head — keep it lean for 3 classes
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False,
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs, name='resnet_scratch')


# ── MixUp (applied per-batch inside a generator wrapper) ────────────────────
def mixup_generator(generator, alpha=0.2):
    """
    Wrap any Keras generator and apply MixUp every batch.
    alpha=0.2 is mild — adjust down to 0.1 if val accuracy drops.
    """
    while True:
        images, labels = next(generator)
        if len(images) < 2:
            yield images, labels
            continue

        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(images))

        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]

        yield mixed_images, mixed_labels


# ── Data generators ──────────────────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    def _flow(datagen, path, shuffle=True):
        return datagen.flow_from_directory(
            path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=shuffle
        )

    print("Loading data generators...")

    if not os.path.exists(DATA_TRAIN):
        raise FileNotFoundError(f"Training data not found at '{DATA_TRAIN}'")

    train_gen = _flow(train_datagen, DATA_TRAIN)

    # Prefer a dedicated val folder; fall back to test
    val_path = DATA_VAL if os.path.exists(DATA_VAL) else DATA_TEST
    if not os.path.exists(val_path):
        raise FileNotFoundError("No val or test folder found.")
    val_gen = _flow(val_datagen, val_path, shuffle=False)

    print(f"  Classes found: {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val   samples : {val_gen.samples}\n")

    return train_gen, val_gen


# ── Learning rate schedule ───────────────────────────────────────────────────
def cosine_warmup_schedule(epoch):
    """
    5-epoch linear warmup → cosine decay.
    Single scheduler — ReduceLROnPlateau removed to avoid conflict.
    """
    initial_lr = 1e-3
    min_lr     = 1e-6
    warmup     = 5

    if epoch < warmup:
        return initial_lr * (epoch + 1) / warmup

    progress = (epoch - warmup) / max(EPOCHS - warmup, 1)
    cosine   = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  Cat / Dog / Other — Training from Scratch")
    print("=" * 55)

    validate_and_clean_images()

    train_gen, val_gen = build_generators()

    # Compute class weights to handle imbalance (other is ~5x smaller)
    labels  = train_gen.classes
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(weights))
    print("Class weights:")
    for idx, name in enumerate(train_gen.class_indices):
        print(f"  {name}: {class_weight_dict[idx]:.3f}")
    print()

    # Wrap training generator with MixUp
    train_with_mixup = mixup_generator(train_gen, alpha=0.2)
    steps_per_epoch  = train_gen.samples // BATCH_SIZE

    model = create_model()

    model.compile(
        optimizer=Adam(learning_rate=1e-3, weight_decay=1e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.05),  # mild smoothing
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    print("\nModel Summary:")
    model.summary()
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        LearningRateScheduler(cosine_warmup_schedule, verbose=0),

        EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            BEST_MODEL,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Kept as a safety net ONLY — cosine schedule handles most of the decay
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("Starting training...\n")
    history = model.fit(
        train_with_mixup,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(FINAL_MODEL)
    print(f"\nBest  model -> {BEST_MODEL}")
    print(f"Final model -> {FINAL_MODEL}")

    # Quick summary
    best_epoch = int(np.argmax(history.history['val_accuracy']))
    best_acc   = history.history['val_accuracy'][best_epoch]
    print(f"\nBest val accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")

    return history, model


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    history, model = train()