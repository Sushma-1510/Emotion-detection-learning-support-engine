# src/preprocessing.py

import os
import re
import pickle
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------
# Download Required NLTK Resources
# ------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80

MODEL_DIR = "models/bltsm"

EMOTIONS = [
    "Bored",
    "Confident",
    "Confused",
    "Curious",
    "Frustrated"
]

# ------------------------------------------------------------------
# Stopwords
# ------------------------------------------------------------------
stop_words = set(stopwords.words("english"))

# Keep important negation words
keep_words = {
    "not",
    "no",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "very",
    "really"
}

stop_words = stop_words - keep_words

# ------------------------------------------------------------------
# Text Cleaning
# ------------------------------------------------------------------
def clean_text(text):
    """
    Clean and normalize text.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove emails
    4. Remove HTML
    5. Remove unwanted characters
    6. Remove stopwords
    7. Normalize whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Keep letters, apostrophes and spaces
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra apostrophes
    text = re.sub(r"\'+", "'", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------
def load_custom_dataset(path="data/emotion_text_dataset.xlsx"):
    """
    Load dataset from Excel or CSV.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    required_columns = {"text", "emotion"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
        )

    df = df.dropna(subset=["text", "emotion"])

    df["text"] = df["text"].astype(str)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Total Samples : {len(df)}")
    print(df["emotion"].value_counts())
    print("=" * 50)

    return df

# ------------------------------------------------------------------
# Preprocessing Pipeline
# ------------------------------------------------------------------
def preprocess(
    df,
    tokenizer=None,
    label_encoder=None,
    fit=True
):
    """
    Complete preprocessing pipeline.

    Returns:
        X_padded
        y
        tokenizer
        label_encoder
    """

    df = df.copy()

    df["clean_text"] = df["text"].apply(clean_text)

    # -----------------------------
    # Label Encoding
    # -----------------------------
    if label_encoder is None:
        label_encoder = LabelEncoder()

    if fit:
        y = label_encoder.fit_transform(df["emotion"])
    else:
        y = label_encoder.transform(df["emotion"])

    # -----------------------------
    # Tokenizer
    # -----------------------------
    if tokenizer is None:
        tokenizer = Tokenizer(
            num_words=MAX_VOCAB_SIZE,
            oov_token="<OOV>"
        )

    if fit:
        tokenizer.fit_on_texts(df["clean_text"])

    sequences = tokenizer.texts_to_sequences(
        df["clean_text"]
    )

    X_padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )

    return (
        X_padded,
        y,
        tokenizer,
        label_encoder
    )

# ------------------------------------------------------------------
# Train / Validation / Test Split
# ------------------------------------------------------------------
def split_data(
    X,
    y,
    val_size=0.10,
    test_size=0.10,
    random_state=42
):
    """
    Split data into:
    Train 80%
    Validation 10%
    Test 10%
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=y,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    print("=" * 50)
    print("Dataset Split")
    print("=" * 50)
    print(f"Train : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test : {len(X_test)}")
    print("=" * 50)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

# ------------------------------------------------------------------
# Save Artifacts
# ------------------------------------------------------------------
def save_artifacts(
    tokenizer,
    label_encoder,
    path=MODEL_DIR
):
    """
    Save tokenizer and label encoder.
    """

    os.makedirs(path, exist_ok=True)

    tokenizer_path = os.path.join(
        path,
        "tokenizer.pkl"
    )

    encoder_path = os.path.join(
        path,
        "label_encoder.pkl"
    )

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print("Artifacts Saved Successfully")

# ------------------------------------------------------------------
# Load Artifacts
# ------------------------------------------------------------------
def load_artifacts(path=MODEL_DIR):
    """
    Load tokenizer and label encoder.
    """

    tokenizer_path = os.path.join(
        path,
        "tokenizer.pkl"
    )

    encoder_path = os.path.join(
        path,
        "label_encoder.pkl"
    )

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(tokenizer_path)

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(encoder_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    print("Artifacts Loaded Successfully")

    return tokenizer, label_encoder

# ------------------------------------------------------------------
# Quick Test
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("Testing clean_text()")
    print("=" * 50)

    samples = [
        "I have NO idea what recursion means!!",
        "I'm so frustrated I can't fix this bug https://stackoverflow.com",
        "Finally understood neural networks, feeling confident!",
        None
    ]

    for sample in samples:
        print("Input :", sample)
        print("Output:", clean_text(sample))
        print()

    print("=" * 50)

    try:
        tokenizer, encoder = load_artifacts()

        print("Classes:")
        print(encoder.classes_)

        print()

        print("Vocabulary Size:")
        print(len(tokenizer.word_index))

    except Exception as e:
        print(e)