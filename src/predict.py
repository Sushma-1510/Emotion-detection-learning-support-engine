# src/predict.py
import numpy as np
import pickle
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ──────────────────────────────────────────────
MAX_SEQ_LEN = 80
EMOTIONS    = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

# ── Keyword weights for mock/rule-based prediction ─────────
KEYWORD_WEIGHTS = {
    'Bored': [
        'boring', 'bored', 'tired', 'same', 'repetitive', 'dull',
        'again', 'already', 'basic', 'know', 'repeat', 'nothing new',
        'over and over', 'sleep', 'tedious', 'monotonous'
    ],
    'Confident': [
        'understand', 'got it', 'finally', 'solved', 'makes sense',
        'clear', 'easy', 'figured', 'done', 'completed', 'success',
        'excellent', 'perfect', 'great', 'amazing', 'mastered'
    ],
    'Confused': [
        'confused', 'confusing', 'lost', 'no idea', 'dont understand',
        "don't understand", 'unclear', 'what does', 'what is', 'how does',
        'help', 'explain', 'makes no sense', "doesn't make sense",
        'never understood', 'cannot understand', 'struggle'
    ],
    'Curious': [
        'wonder', 'curious', 'how', 'why', 'what if', 'interesting',
        'explore', 'want to know', 'learn more', 'tell me', 'fascinating',
        'cool', 'amazing', 'discover', 'investigate'
    ],
    'Frustrated': [
        'frustrated', 'frustrating', 'annoying', 'cant', "can't",
        'stuck', 'hours', 'keep failing', 'wrong', 'give up', 'useless',
        'hate', 'bug', 'error', 'broken', 'not working', 'mad',
        'angry', 'terrible'
    ]
}


# ── Keyword-Based Mock Predictor ────────────────────────────
def _keyword_predict(text: str) -> np.ndarray:
    """
    Rule-based emotion prediction using keyword scoring.
    Used as fallback when trained models are unavailable.
    """
    text_lower = text.lower()
    text_lower = re.sub(r'[^a-z\s]', ' ', text_lower)

    scores = {e: 0.0 for e in EMOTIONS}

    for emotion, keywords in KEYWORD_WEIGHTS.items():
        for kw in keywords:
            if kw in text_lower:
                if kw == emotion.lower():
                    scores[emotion] += 8.0
                elif len(kw.split()) > 1:
                    scores[emotion] += 6.0
                else:
                    scores[emotion] += 3.0

    if any(word in text_lower for word in ['but', 'though', 'however']):
        if any(word in text_lower for word in ['love', 'enjoy', 'happy', 'excited']):
            scores['Confident'] += 2.0
        if any(word in text_lower for word in ['struggling', 'confused', 'lost', 'hard', 'difficult']):
            scores['Confused'] += 2.0
            scores['Frustrated'] += 1.0

    if sum(scores.values()) == 0:
        # Fallback heuristic for otherwise unlabeled or neutral text
        if any(word in text_lower for word in [
                'struggling', 'hard', 'difficult', 'challenging',
                'confusing', 'lost', 'unclear', 'help', 'stuck',
                'problem', 'issue', 'cannot', 'dont', 'doesnt']):
            scores['Confused'] += 8.0
            scores['Frustrated'] += 5.0

        if any(word in text_lower for word in [
                'love', 'enjoy', 'like', 'excited', 'great', 'good',
                'happy', 'amazing', 'awesome', 'nice', 'best']):
            scores['Confident'] += 6.0
            scores['Curious'] += 3.0

        if any(word in text_lower for word in [
                'wonder', 'why', 'how', 'what', 'explore', 'learn more',
                'curious', 'interested', 'discover']):
            scores['Curious'] += 8.0

        if any(word in text_lower for word in [
                'boring', 'bored', 'tired', 'same', 'repetitive', 'dull',
                'sleepy', 'unmotivated', 'meh', 'blah']):
            scores['Bored'] += 8.0

        if any(word in text_lower for word in [
                'frustrated', 'annoyed', 'angry', 'give up', 'useless',
                'broken', 'wrong', 'error', 'not working', 'hate',
                'terrible', 'stress', 'anxiety']):
            scores['Frustrated'] += 8.0

        if sum(scores.values()) == 0:
            scores['Curious'] += 3.0
            scores['Confused'] += 2.0
            scores['Confident'] += 2.0
            scores['Bored'] += 1.0
            scores['Frustrated'] += 1.0

    # Smooth final probabilities so no single emotion dominates too strongly.
    vals = np.array([scores[e] + 1.0 for e in EMOTIONS], dtype=float)
    exp_vals = np.exp(vals)
    probs = exp_vals / exp_vals.sum()
    return probs


# ── Try to Load Real Models ─────────────────────────────────
_bilstm        = None
_tokenizer     = None
_le            = None
_bert          = None
_bert_tokenizer= None
_device        = None
_models_loaded = False


def _try_load_models():
    """Attempt to load trained models; silently skip if missing."""
    global _bilstm, _tokenizer, _le, _bert, _bert_tokenizer, _device, _models_loaded

    if _models_loaded:
        return

    # ── BiLSTM ──────────────────────────────────────────────
    try:
        from src.preprocessing import load_artifacts
        from src.model import load_bilstm

        bilstm_path = 'models/bltsm/bilstm_final.keras'
        art_path    = 'models/bltsm/'

        if (os.path.exists(os.path.join(art_path, 'tokenizer.pkl')) and
                os.path.exists(bilstm_path)):
            _tokenizer, _le = load_artifacts(art_path)
            _bilstm         = load_bilstm(bilstm_path)
            print("BiLSTM model loaded")
        else:
            print("BiLSTM model files not found — using keyword fallback")
    except Exception as e:
        print(f"BiLSTM load skipped: {e}")

    # ── BERT ────────────────────────────────────────────────
    try:
        from src.bert_model import load_bert

        bert_path = 'models/bert_emotion_model_final/bert_model.pt'

        if os.path.exists(bert_path):
            _bert, _bert_tokenizer, _device = load_bert(bert_path)
            print("BERT model loaded")
        else:
            print("BERT model file not found — using keyword fallback")
    except Exception as e:
        print(f"BERT load skipped: {e}")

    _models_loaded = True


# ── Runtime Load Helpers ───────────────────────────────────
def _save_bytes_to_path(bobj, path: str):
    """Save a file-like or bytes object to disk."""
    dirp = os.path.dirname(path)
    if dirp and not os.path.exists(dirp):
        os.makedirs(dirp, exist_ok=True)

    # bobj may be bytes or file-like
    data = None
    if isinstance(bobj, (bytes, bytearray)):
        data = bobj
    else:
        try:
            # streamlit gives UploadedFile which has .read()
            data = bobj.read()
        except Exception:
            raise ValueError("Unsupported file-like object")

    with open(path, "wb") as f:
        f.write(data)


def load_bilstm_runtime(model_file=None, tokenizer_file=None, label_encoder_file=None):
    """Load a BiLSTM model and its artifacts at runtime from uploaded files.

    Args:
        model_file: bytes or file-like for the Keras model (.keras/.h5 or saved model archive)
        tokenizer_file: bytes or file-like for `tokenizer.pkl`
        label_encoder_file: bytes or file-like for `label_encoder.pkl`

    Returns: True if model loaded successfully, otherwise raises.
    """
    global _bilstm, _tokenizer, _le, _models_loaded

    from src.model import load_bilstm
    from src.preprocessing import load_artifacts

    art_dir = 'models/bltsm'
    model_path = os.path.join(art_dir, 'bilstm_final.keras')

    if model_file is not None:
        _save_bytes_to_path(model_file, model_path)

    if tokenizer_file is not None:
        _save_bytes_to_path(tokenizer_file, os.path.join(art_dir, 'tokenizer.pkl'))

    if label_encoder_file is not None:
        _save_bytes_to_path(label_encoder_file, os.path.join(art_dir, 'label_encoder.pkl'))

    # Attempt to load
    _tokenizer, _le = load_artifacts(art_dir)
    _bilstm = load_bilstm(model_path)
    _models_loaded = True
    return True


def load_bert_runtime(model_file=None):
    """Load a fine-tuned BERT model at runtime from an uploaded `.pt` file.

    Args:
        model_file: bytes or file-like containing PyTorch state_dict

    Returns: True if loaded successfully.
    """
    global _bert, _bert_tokenizer, _device, _models_loaded

    from src.bert_model import load_bert

    bert_dir = 'models/bert_emotion_model_final'
    model_path = os.path.join(bert_dir, 'bert_model.pt')

    if model_file is None:
        raise ValueError("No model_file provided")

    _save_bytes_to_path(model_file, model_path)

    _bert, _bert_tokenizer, _device = load_bert(model_path)
    _models_loaded = True
    return True


def reload_models_if_present():
    """Convenience: re-run the model discovery to load any models now present on disk."""
    # Reset flags so _try_load_models will attempt to load again
    global _models_loaded
    _models_loaded = False
    _try_load_models()


# Load models on import (non-crashing)
_try_load_models()


# ── BiLSTM Inference ────────────────────────────────────────
def _bilstm_predict(text: str) -> np.ndarray:
    """Get emotion probabilities from BiLSTM or fallback."""
    if _bilstm is None or _tokenizer is None or _le is None:
        return _keyword_predict(text)

    try:
        from src.preprocessing import clean_text
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = clean_text(text)
        seq     = _tokenizer.texts_to_sequences([cleaned])
        padded  = pad_sequences(seq, maxlen=MAX_SEQ_LEN,
                                 padding='post', truncating='post')
        probs = _bilstm.predict(padded, verbose=0)[0]
        return probs
    except Exception as e:
        print(f"BiLSTM inference error: {e} — using fallback")
        return _keyword_predict(text)


# ── BERT Inference ──────────────────────────────────────────
def _bert_infer(text: str) -> np.ndarray:
    """Get emotion probabilities from BERT or fallback."""
    if _bert is None or _bert_tokenizer is None:
        return _keyword_predict(text)

    try:
        from src.bert_model import bert_predict
        return bert_predict(text, _bert, _bert_tokenizer, _device)
    except Exception as e:
        print(f"BERT inference error: {e} — using fallback")
        return _keyword_predict(text)


# ── Label Classes Helper ────────────────────────────────────
def _get_classes():
    """Return label classes (real or fallback)."""
    if _le is not None:
        return list(_le.classes_)
    return EMOTIONS


def is_fallback_mode() -> bool:
    """Return True if any trained model is unavailable."""
    return (_bilstm is None or _tokenizer is None or _le is None or
            _bert is None or _bert_tokenizer is None)


# ── Mixed Emotion Detection ─────────────────────────────────
def _get_mixed_emotions(probs: np.ndarray,
                         classes: list,
                         threshold: float = 0.15) -> list:
    """Return emotions above threshold; always include top 2 if close."""
    top_idx = int(np.argmax(probs))
    sorted_idxs = np.argsort(probs)[::-1]
    mixed = [classes[top_idx]]

    if len(sorted_idxs) > 1:
        second_idx = int(sorted_idxs[1])
        if second_idx != top_idx and probs[second_idx] >= probs[top_idx] * 0.25:
            mixed.append(classes[second_idx])

    for i, p in enumerate(probs):
        emotion = classes[i]
        if p >= threshold and emotion not in mixed:
            mixed.append(emotion)

    return mixed if mixed else [classes[top_idx]]


# ── Core Prediction Function ────────────────────────────────
def predict_emotion(text: str, model: str = "bert") -> dict:
    """
    Predict emotion from student text input.

    Args:
        text  : student's text input
        model : "bert" or "bilstm"

    Returns:
        {
            "emotion"       : "Confused",
            "confidence"    : 0.87,
            "all_scores"    : { "Bored": 0.03, ... },
            "mixed_emotions": ["Confused", "Curious"]
        }
    """
    if not text or not text.strip():
        return {
            "emotion"       : "Confused",
            "confidence"    : 0.0,
            "all_scores"    : {e: 0.0 for e in EMOTIONS},
            "mixed_emotions": ["Confused"]
        }

    classes = _get_classes()

    # Get probabilities from selected model
    if model.lower() == "bert":
        probs = _bert_infer(text)
    else:
        probs = _bilstm_predict(text)

    # Ensure probs has the right length
    if len(probs) != len(classes):
        probs = _keyword_predict(text)
        classes = EMOTIONS

    top_idx    = int(np.argmax(probs))
    emotion    = classes[top_idx]
    confidence = float(probs[top_idx])
    all_scores = {
        classes[i]: round(float(probs[i]), 4)
        for i in range(len(classes))
    }
    mixed = _get_mixed_emotions(probs, classes)

    return {
        "emotion"       : emotion,
        "confidence"    : round(confidence, 4),
        "all_scores"    : all_scores,
        "mixed_emotions": mixed
    }


# ── Compare Both Models ─────────────────────────────────────
def compare_models(text: str) -> dict:
    """
    Run both BiLSTM and BERT on same text.
    Returns both results for side-by-side comparison.
    """
    bert_result   = predict_emotion(text, model="bert")
    bilstm_result = predict_emotion(text, model="bilstm")

    return {
        "text"     : text,
        "bert"     : bert_result,
        "bilstm"   : bilstm_result,
        "agreement": bert_result["emotion"] == bilstm_result["emotion"]
    }


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing predict_emotion()")
    print("=" * 60)

    tests = [
        ("I have no idea what recursion means no matter how many times I read it", "Confused"),
        ("I finally understand how neural networks work and it all makes sense",    "Confident"),
        ("I have been stuck on this bug for five hours and want to give up",        "Frustrated"),
        ("I wonder how transformers handle very long documents",                    "Curious"),
        ("This lecture is just repeating last week with nothing new",               "Bored"),
    ]

    print("\n--- BERT Results ---")
    for text, expected in tests:
        result = predict_emotion(text, model="bert")
        match  = "✅" if result["emotion"] == expected else "❌"
        print(f"{match} Expected: {expected:12} Got: {result['emotion']:12} Conf: {result['confidence']:.2f}")

    print("\n--- BiLSTM Results ---")
    for text, expected in tests:
        result = predict_emotion(text, model="bilstm")
        match  = "✅" if result["emotion"] == expected else "❌"
        print(f"{match} Expected: {expected:12} Got: {result['emotion']:12} Conf: {result['confidence']:.2f}")