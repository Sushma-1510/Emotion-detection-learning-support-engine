# src/bert_model.py

import os
import numpy as np
import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MAX_SEQ_LEN = 80
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_NAME = "bert-base-uncased"

# -------------------------------------------------------
# Load BERT Model
# -------------------------------------------------------

def load_bert(
    model_path="models/bert_emotion_model_final/bert_model.pt"
):
    """
    Load fine-tuned BERT model.

    Returns
    -------
    model
    tokenizer
    device
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nBERT model not found:\n{model_path}"
        )

    print("=" * 60)
    print("Loading BERT Model...")
    print("=" * 60)

    tokenizer = BertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print("BERT Loaded Successfully")
    print(f"Running on: {DEVICE}")
    print("=" * 60)

    return model, tokenizer, DEVICE


# -------------------------------------------------------
# Predict
# -------------------------------------------------------

def bert_predict(
    text,
    model,
    tokenizer,
    device,
    max_len=MAX_SEQ_LEN
):
    """
    Returns probability vector.
    """

    if text is None:
        text = ""

    text = str(text).strip()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probs = torch.softmax(
            logits,
            dim=1
        )

    return probs.squeeze().cpu().numpy()


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    emotions = [
        "Bored",
        "Confident",
        "Confused",
        "Curious",
        "Frustrated"
    ]

    model, tokenizer, device = load_bert()

    tests = [

        "I don't understand recursion",

        "This lecture is boring",

        "I finally solved the bug",

        "I am curious about transformers",

        "This assignment is making me frustrated"

    ]

    print()

    for text in tests:

        probs = bert_predict(
            text,
            model,
            tokenizer,
            device
        )

        idx = np.argmax(probs)

        print("=" * 60)
        print("Input :", text)
        print("Emotion :", emotions[idx])
        print("Confidence :", round(float(probs[idx]), 4))