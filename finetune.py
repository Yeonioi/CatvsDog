"""
Cat vs Dog vs Other — Fine-tuning Script (Improved)
Transfer learning with:
  - Staged/gradual unfreezing
  - BN layers always frozen
  - Cosine LR decay (matches train.py)
  - EarlyStopping restored
  - MixUp re-added
  - Safe recompile between phases
"""

import os
import numpy as np

# ── CPU config ───────────────────────────────────────────────────────────────
os.environ['TF_NUM_INTRAOP_THREADS'] = '10'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as klayers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE         = 128
BATCH_SIZE       = 32
NUM_CLASSES      = 3
MODEL_DIR        = 'models'
PRETRAINED_MODEL = os.path.join(MODEL_DIR, 'best_model.keras')
FINETUNED_MODEL  = os.path.join(MODEL_DIR, 'finetuned_model.keras')
DATA_TRAIN       = 'data/train'
DATA_VAL         = 'data/val'
DATA_TEST        = 'data/test'

# Phase 1: head-only warmup
PHASE1_EPOCHS    = 5
PHASE1_LR        = 1e-3     # Higher LR — only the new head is training

# Phase 2: gradual unfreeze of later stages
PHASE2_EPOCHS    = 15
PHASE2_LR        = 1e-4     # 10x lower — touching pretrained weights

# Phase 3: deeper unfreeze
PHASE3_EPOCHS    = 15
PHASE3_LR        = 3e-5     # Even lower — going deeper into the network

TOTAL_EPOCHS     = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS  # 35


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


# ── Freezing helpers ─────────────────────────────────────────────────────────
def freeze_all(model):
    """Freeze every layer including BN."""
    for layer in model.layers:
        layer.trainable = False


def unfreeze_from(model, from_index):
    """
    Unfreeze layers from `from_index` onward — but NEVER unfreeze BN layers.
    BN statistics trained on ImageNet (or your base training set) are
    valuable; re-training them with small fine-tuning batches destabilises
    the network.
    """
    for layer in model.layers[from_index:]:
        if isinstance(layer, klayers.BatchNormalization):
            layer.trainable = False   # Always keep BN frozen
        else:
            layer.trainable = True

    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"  Trainable layers: {trainable} / {total}  "
          f"(BN layers always frozen)\n")


# ── LR schedules ─────────────────────────────────────────────────────────────
def make_cosine_schedule(base_lr, total_epochs, warmup=2):
    """
    Returns a LearningRateScheduler-compatible function.
    `epoch` here is the LOCAL epoch within a phase (reset each phase).
    """
    min_lr = base_lr * 0.01

    def schedule(epoch):
        if epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    return schedule


# ── MixUp generator ───────────────────────────────────────────────────────────
def mixup_generator(generator, alpha=0.2, class_weight_dict=None):
    """
    Wrap any Keras generator and apply MixUp every batch.

    class_weight is baked into per-sample weights returned as the third
    element of the tuple — the only way to handle weighting with a Python
    generator (Keras rejects the class_weight= kwarg for generators).
    """
    while True:
        images, labels = next(generator)
        if len(images) < 2:
            if class_weight_dict is not None:
                class_indices = np.argmax(labels, axis=1)
                sample_weights = np.array([class_weight_dict[c] for c in class_indices],
                                          dtype=np.float32)
                yield images, labels, sample_weights
            else:
                yield images, labels
            continue

        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(images))

        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]

        if class_weight_dict is not None:
            # Blend sample weights the same way labels are blended
            class_indices_a = np.argmax(labels, axis=1)
            class_indices_b = np.argmax(labels[idx], axis=1)
            weights_a = np.array([class_weight_dict[c] for c in class_indices_a],
                                  dtype=np.float32)
            weights_b = np.array([class_weight_dict[c] for c in class_indices_b],
                                  dtype=np.float32)
            sample_weights = lam * weights_a + (1 - lam) * weights_b
            yield mixed_images, mixed_labels, sample_weights
        else:
            yield mixed_images, mixed_labels


# ── Data generators ───────────────────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
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

    val_path = DATA_VAL if os.path.exists(DATA_VAL) else DATA_TEST
    if not os.path.exists(val_path):
        raise FileNotFoundError("No val or test folder found.")
    val_gen = _flow(val_datagen, val_path, shuffle=False)

    print(f"  Classes found : {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val   samples : {val_gen.samples}\n")

    return train_gen, val_gen


# ── Compile helper ────────────────────────────────────────────────────────────
def compile_model(model, lr):
    """
    Always recompile after changing layer.trainable flags.
    Keras requires this so the new trainable mask is reflected in the graph.
    """
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )


# ── Callbacks factory ─────────────────────────────────────────────────────────
def make_callbacks(phase_epochs, base_lr, save_path=None):
    cbs = [
        LearningRateScheduler(
            make_cosine_schedule(base_lr, phase_epochs), verbose=0
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,           # Generous — fine-tuning can stall temporarily
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    if save_path:
        cbs.append(ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ))
    return cbs


# ── Main fine-tuning routine ──────────────────────────────────────────────────
def finetune():
    print("=" * 65)
    print("  Cat / Dog / Other — Fine-tuning (Gradual Unfreeze)")
    print("=" * 65)
    print()

    if not os.path.exists(PRETRAINED_MODEL):
        print(f"ERROR: Pretrained model not found at '{PRETRAINED_MODEL}'")
        print("Run train.py first to create the base model.")
        return

    print(f"Loading pretrained model: {PRETRAINED_MODEL}")
    model = keras.models.load_model(PRETRAINED_MODEL)
    print(f"Total layers: {len(model.layers)}\n")

    validate_and_clean_images()
    train_gen, val_gen = build_generators()

    labels           = train_gen.classes
    weights          = compute_class_weight('balanced',
                                            classes=np.unique(labels),
                                            y=labels)
    class_weight_dict = dict(enumerate(weights))
    print("Class weights:")
    for idx, name in enumerate(train_gen.class_indices):
        print(f"  {name}: {class_weight_dict[idx]:.3f}")
    print()

    # Class weights are baked into the generator as sample_weight (3rd tuple
    # element) — Keras does not support class_weight= for Python generators.
    train_with_mixup = mixup_generator(train_gen, alpha=0.2,
                                       class_weight_dict=class_weight_dict)
    steps_per_epoch  = train_gen.samples // BATCH_SIZE

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── PHASE 1: Head only ────────────────────────────────────────────────────
    print("─" * 65)
    print(f"PHASE 1 — Head warmup ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    print("  Training only the final Dense layers; all base layers frozen.")
    print("─" * 65)

    freeze_all(model)
    # Unfreeze only the last Dense + BN head (last 4 layers typically)
    unfreeze_from(model, from_index=-4)
    compile_model(model, PHASE1_LR)

    history1 = model.fit(
        train_with_mixup,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(PHASE1_EPOCHS, PHASE1_LR),
        verbose=1
    )

    # ── PHASE 2: Unfreeze later stages ───────────────────────────────────────
    print()
    print("─" * 65)
    print(f"PHASE 2 — Later stages ({PHASE2_EPOCHS} epochs, LR={PHASE2_LR})")
    print("  Unfreezing last ~25% of layers (Stage 4 + head).")
    print("─" * 65)

    unfreeze_from(model, from_index=int(len(model.layers) * 0.75))
    compile_model(model, PHASE2_LR)   # MUST recompile after trainable change

    history2 = model.fit(
        train_with_mixup,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        initial_epoch=0,
        callbacks=make_callbacks(PHASE2_EPOCHS, PHASE2_LR),
        verbose=1
    )

    # ── PHASE 3: Deeper unfreeze ──────────────────────────────────────────────
    print()
    print("─" * 65)
    print(f"PHASE 3 — Deeper unfreeze ({PHASE3_EPOCHS} epochs, LR={PHASE3_LR})")
    print("  Unfreezing last ~50% of layers (Stages 3–4 + head).")
    print("─" * 65)

    unfreeze_from(model, from_index=int(len(model.layers) * 0.50))
    compile_model(model, PHASE3_LR)   # Recompile again

    history3 = model.fit(
        train_with_mixup,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=PHASE3_EPOCHS,
        initial_epoch=0,
        callbacks=make_callbacks(PHASE3_EPOCHS, PHASE3_LR,
                                 save_path=FINETUNED_MODEL),
        verbose=1
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Fine-tuning Complete!")
    print("=" * 65)
    print(f"Best finetuned model saved to: {FINETUNED_MODEL}")
    print()

    all_val_acc = (
        history1.history['val_accuracy'] +
        history2.history['val_accuracy'] +
        history3.history['val_accuracy']
    )
    best_epoch = int(np.argmax(all_val_acc)) + 1
    best_acc   = float(np.max(all_val_acc))
    print(f"Best val accuracy across all phases: {best_acc:.4f} "
          f"(epoch {best_epoch} overall)")
    print()
    print("Next steps:")
    print("  1. Confirm app.py MODEL_PATH points to 'models/finetuned_model.keras'")
    print("  2. Run predictions on a held-out test set")
    print("  3. If results are good, promote finetuned_model.keras → best_model.keras")
    print()


if __name__ == '__main__':
    finetune()