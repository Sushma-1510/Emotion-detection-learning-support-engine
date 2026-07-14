# logger.py
import os
import sys
import csv
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Fix Windows encoding for print statements
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# -- Constants ------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(DATA_DIR, 'interactions_log.csv')

COLUMNS = [
    'timestamp',
    'student_text',
    'emotion',
    'confidence',
    'mixed_emotions',
    'bilstm_emotion',
    'bilstm_confidence',
    'bert_emotion',
    'bert_confidence',
    'gemini_response',
    'model_used'
]


# -- Initialize Log File --------------------------------
def init_log():
    """Create log file with headers if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


# -- Log Single Interaction -----------------------------
def log_interaction(student_text: str,
                    emotion: str,
                    confidence: float,
                    gemini_response: str,
                    mixed_emotions: list = None,
                    bilstm_emotion: str = "",
                    bilstm_confidence: float = 0.0,
                    bert_emotion: str = "",
                    bert_confidence: float = 0.0,
                    model_used: str = "bert") -> bool:
    """Log one student interaction to CSV."""
    init_log()
    try:
        row = {
            'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'student_text':      str(student_text)[:500],
            'emotion':           emotion,
            'confidence':        round(float(confidence), 4),
            'mixed_emotions':    str(mixed_emotions or []),
            'bilstm_emotion':    bilstm_emotion,
            'bilstm_confidence': round(float(bilstm_confidence), 4),
            'bert_emotion':      bert_emotion,
            'bert_confidence':   round(float(bert_confidence), 4),
            'gemini_response':   str(gemini_response)[:1000],
            'model_used':        model_used
        }
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False


# -- Read All Logs --------------------------------------
def get_all_logs() -> pd.DataFrame:
    """Read all logged interactions."""
    init_log()
    try:
        df = pd.read_csv(
            LOG_FILE,
            encoding='utf-8',
            on_bad_lines='skip',
            dtype=str
        )
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ''
        if 'confidence' in df.columns:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        print(f"Error reading logs: {e}")
        return pd.DataFrame(columns=COLUMNS)


# -- Get Stats ------------------------------------------
def get_stats() -> dict:
    """Get summary statistics from logs."""
    df = get_all_logs()
    if df.empty or 'emotion' not in df.columns:
        return {
            'total_sessions':  0,
            'emotion_counts':  {},
            'avg_confidence':  0.0,
            'most_common':     'None',
            'recent_sessions': []
        }

    try:
        emotion_counts = df['emotion'].fillna('None').value_counts().to_dict()
        most_common = next((k for k, _ in emotion_counts.items() if k and k != 'None'), 'None')
        avg_confidence = float(df['confidence'].mean()) if 'confidence' in df.columns else 0.0
        return {
            'total_sessions':  len(df),
            'emotion_counts':  emotion_counts,
            'avg_confidence':  round(avg_confidence, 4),
            'most_common':     most_common,
            'recent_sessions': df.tail(5).to_dict('records')
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            'total_sessions':  0,
            'emotion_counts':  {},
            'avg_confidence':  0.0,
            'most_common':     'None',
            'recent_sessions': []
        }


# -- Delete Log Entry -----------------------------------
def delete_log_entry(index: int) -> bool:
    """Delete a specific log entry by index."""
    try:
        df = get_all_logs()
        if index < 0 or index >= len(df):
            return False
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(LOG_FILE, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Delete error: {e}")
        return False


# -- Clear All Logs -------------------------------------
def clear_all_logs() -> bool:
    """Clear all logged interactions."""
    try:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
        print("All logs cleared")
        return True
    except Exception as e:
        print(f"Clear error: {e}")
        return False


if __name__ == "__main__":
    print("Testing logger.py...")
    success = log_interaction(
        student_text="I have no idea what recursion means",
        emotion="Confused",
        confidence=0.87,
        gemini_response="Try breaking it into smaller steps...",
        mixed_emotions=["Confused", "Frustrated"],
        bilstm_emotion="Confused",
        bilstm_confidence=1.0,
        bert_emotion="Bored",
        bert_confidence=0.34,
        model_used="bert"
    )
    print(f"Logged: {success}")
    stats = get_stats()
    print(f"Stats: {stats}")
    print("Logger OK")