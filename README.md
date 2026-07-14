# AI-Driven Emotion Detection & Personalized Learning Support Platform

The AI-Driven Emotion Detection & Personalized Learning Support Platform is an end-to-end system that transforms a learner’s free-text description of their study challenge into an emotion-aware, supportive response. By combining deep learning emotion classifiers (BiLSTM and BERT) with rule-based keyword enhancement and Gemini-generated guidance, the platform ingests raw student input, predicts nuanced emotional states (Bored, Confident, Confused, Curious, Frustrated), and delivers tailored tips, next steps, and encouragement.

## Features
- **Emotion Classification**: Uses BiLSTM and BERT models to detect emotions.
- **Mixed Emotion Detection**: Highlights mixed emotions with a confidence threshold.
- **Personalized Responses**: Generates AI responses using Gemini 2.5 Flash, with template fallbacks.
- **Streamlit Dashboard**: A beautiful, interactive dashboard with model comparison and analytics.

## Setup
1. Create a virtual environment (`python -m venv venv`)
2. Install requirements (`pip install -r requirements.txt`)
3. Add your Gemini API Key in the `.env` file.
4. Run the app (`streamlit run app.py`)
