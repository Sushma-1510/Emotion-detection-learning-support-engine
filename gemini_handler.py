# gemini_handler.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configure Gemini (modern google-genai package) ──────────
_api_key = os.getenv("GEMINI_API_KEY", "")
_client  = None

def _get_client():
    """Lazy-initialize the Gemini client."""
    global _client
    if _client is not None:
        return _client
    if not _api_key:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=_api_key)
        return _client
    except ImportError:
        # Fallback to legacy package
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=_api_key)
            return "legacy"
        except Exception:
            return None
    except Exception:
        return None


def _generate_with_legacy(prompt: str) -> str:
    """Use legacy google.generativeai package."""
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _generate(prompt: str) -> str:
    """Generate text using Gemini API."""
    client = _get_client()
    if client is None:
        raise RuntimeError("No Gemini API key configured")

    if client == "legacy":
        return _generate_with_legacy(prompt)

    # Modern google-genai client
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()


# ── Main Function ───────────────────────────────────────────
def get_gemini_response(emotion: str, text: str) -> str:
    """
    Generate personalized learning response using Gemini AI.

    Args:
        emotion : detected emotion (Confused/Frustrated/etc)
        text    : student's original input text

    Returns:
        personalized guidance string
    """
    prompt = f"""
You are a supportive and empathetic AI learning assistant 
helping a student who is feeling {emotion} about their studies.

The student said: "{text}"

Based on their {emotion} emotional state, provide:
1. A brief empathetic acknowledgment (1-2 sentences)
2. 2-3 specific, actionable study tips for someone feeling {emotion}
3. An encouraging closing statement

Keep your response warm, concise and practical.
Format it as flowing paragraphs, not bullet points.
Maximum 150 words.
"""
    try:
        return _generate(prompt)
    except Exception:
        return _get_fallback_response(emotion)


# ── Emotion Specific Prompts ────────────────────────────────
def get_detailed_response(emotion: str,
                           text: str,
                           subject: str = "") -> str:
    """
    Generate detailed response with subject context.
    """
    subject_context = f"about {subject}" if subject else ""

    prompt = f"""
You are an expert AI tutor helping a student who feels 
{emotion} {subject_context}.

Student's message: "{text}"

Provide a personalized response that includes:
1. Empathetic acknowledgment of their {emotion} state
2. Root cause analysis - why students typically feel {emotion}
3. Three specific strategies to overcome this feeling
4. A concrete next step they can take in the next 10 minutes
5. Motivational closing

Keep it conversational, warm and under 200 words.
"""
    try:
        return _generate(prompt)
    except Exception:
        return _get_fallback_response(emotion)


# ── Fallback Responses ──────────────────────────────────────
def _get_fallback_response(emotion: str) -> str:
    """Fallback responses when Gemini API is unavailable."""
    fallbacks = {
        "Confused": (
            "It's completely normal to feel confused "
            "when learning something new. Try breaking "
            "the concept into smaller parts and tackle "
            "one piece at a time. Draw diagrams, use "
            "analogies, and don't hesitate to look for "
            "alternative explanations. You're making "
            "progress even when it doesn't feel like it!"
        ),
        "Frustrated": (
            "Your frustration shows how much you care "
            "about understanding this. Take a short 10-"
            "minute break to reset your mind. When you "
            "return, try a completely different approach "
            "or ask someone for a fresh perspective. "
            "Every expert has been exactly where you "
            "are right now!"
        ),
        "Curious": (
            "Your curiosity is your greatest learning "
            "superpower! Follow that interest and dive "
            "deeper into the topic. Try building a small "
            "project around what you're exploring or "
            "connect it to real-world applications. "
            "Curious learners become the best innovators!"
        ),
        "Confident": (
            "You're doing amazing! This is the perfect "
            "time to challenge yourself further. Try "
            "teaching this concept to someone else or "
            "tackle the most difficult problems you can "
            "find. Your confidence is well earned — "
            "now push yourself to the next level!"
        ),
        "Bored": (
            "Your brain is clearly ready for more "
            "challenge! Use this opportunity to explore "
            "advanced topics or connect this material "
            "to something you're passionate about. "
            "Try creating your own problems or finding "
            "real-world applications. You're ready "
            "for the next level!"
        )
    }
    return fallbacks.get(emotion, fallbacks["Confused"])


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Gemini Handler...")
    print("=" * 50)

    tests = [
        ("Confused",   "I have no idea what recursion means"),
        ("Frustrated", "I have been stuck on this bug for 5 hours"),
        ("Curious",    "I wonder how neural networks actually learn"),
        ("Confident",  "I finally understand gradient descent"),
        ("Bored",      "This lecture keeps repeating what I know"),
    ]

    for emotion, text in tests:
        print(f"\nEmotion: {emotion}")
        print(f"Input:   {text}")
        response = get_gemini_response(emotion, text)
        print(f"Response: {response}")
        print("-" * 50)