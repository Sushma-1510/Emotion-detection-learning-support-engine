# src/train.py

import os
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from preprocessing import (
    load_custom_dataset,
    preprocess,
    split_data,
    save_artifacts
)

from model import build_bilstm

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_DIR = "models/bltsm"
MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_final.keras")

BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_custom_dataset("data/emotion_text_dataset.xlsx")

# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------
print("\nPreprocessing Dataset...")

X, y, tokenizer, label_encoder = preprocess(df)

save_artifacts(tokenizer, label_encoder)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print("Dataset Ready!")
print()

# ---------------------------------------------------------
# Build Model
# ---------------------------------------------------------
print("Building BiLSTM Model...")

model = build_bilstm()

model.summary()

# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
print("\nTraining Started...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1

)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("=" * 60)
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")
print("=" * 60)

# ---------------------------------------------------------
# Save Final Model
# ---------------------------------------------------------

model.save(MODEL_PATH)

print()
print("Model Saved Successfully!")
print(MODEL_PATH)

print()
print("Training Completed Successfully!")