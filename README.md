# 🎓 Emotion Detection Learning Support Engine

An AI-powered learning support system that detects a student's emotional state from text and provides personalized learning assistance using Machine Learning, Deep Learning, and Google Generative AI.

The system identifies emotions expressed by students and recommends supportive responses, motivational messages, and learning guidance to improve engagement and learning outcomes.

---

# 🚀 Features

* 😊 Detects emotions from user input text
* 🧠 Uses multiple AI models for prediction

  * Traditional Machine Learning
  * BiLSTM Deep Learning Model
  * BERT Transformer Model
* 🤖 AI-generated learning suggestions using Google Gemini
* 📊 Displays predicted emotion with confidence score
* 🎨 Interactive Streamlit web application
* ⚡ Fast real-time prediction
* 📈 Clean and user-friendly interface

---

# 🛠 Tech Stack

### Programming Language

* Python 3.11+

### Machine Learning

* Scikit-learn
* TensorFlow / Keras
* BERT
* NumPy
* Pandas

### NLP

* NLTK
* Tokenization
* Text Cleaning

### Frontend

* Streamlit

### Generative AI

* Google Gemini API

### Visualization

* Matplotlib
* Seaborn

---

# 📂 Project Structure

```
Emotion-detection-learning-support-engine/

│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── models/
│   ├── bert_model
│   ├── bilstm_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── model.py
│   ├── bert_model.py
│   └── utils.py
│
├── notebooks/
│
└── assets/
    └── screenshots
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Sushma-1510/Emotion-detection-learning-support-engine.git
```

Move into the project directory

```bash
cd Emotion-detection-learning-support-engine
```

Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Google Gemini API

Create a `.env` file in the project root.

```
GOOGLE_API_KEY=YOUR_API_KEY
```

Replace `YOUR_API_KEY` with your Google Gemini API key.

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧠 Workflow

```
User Input
      │
      ▼
Text Preprocessing
      │
      ▼
Emotion Prediction
(BERT / BiLSTM / ML)
      │
      ▼
Detected Emotion
      │
      ▼
Google Gemini
      │
      ▼
Learning Support Suggestion
      │
      ▼
Displayed on Streamlit
```

---

# 🎯 Supported Emotions

* Happy
* Sad
* Angry
* Fear
* Surprise
* Love
* Neutral

*(The available emotions depend on the trained dataset.)*

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Sushma-1510/Emotion-detection-learning-support-engine.git
cd Emotion-detection-learning-support-engine
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root and add your Google Gemini API key.

```env
GOOGLE_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
```

---

## 5. Train the Models (Skip if Pre-trained Models Are Available)

If the trained models are not already present in the `models/` directory, run the training scripts.

### Train the Machine Learning Model

```bash
python src/model.py
```

### Train the BiLSTM Model

```bash
python src/bilstm_model.py
```

### Train the BERT Model

```bash
python src/bert_model.py
```

> If your repository already contains the trained model files, you can skip this step.

---

## 6. Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## 7. Open the Application

Once the server starts, open your browser and visit:

```
http://localhost:8501
```

---

# 📋 Usage

1. Open the Streamlit application.
2. Enter a sentence expressing your feelings or emotions.
3. Click the **Predict** button.
4. The system detects the emotion using the selected AI model.
5. Google Gemini generates personalized learning support and motivational suggestions based on the detected emotion.
6. View the predicted emotion, confidence score, and AI-generated response on the screen.

---

# 🛑 Troubleshooting

### ModuleNotFoundError

Install all required packages:

```bash
pip install -r requirements.txt
```

### Missing API Key

Make sure the `.env` file contains a valid Google Gemini API key.

### Streamlit Command Not Found

Run:

```bash
pip install streamlit
```
 
or

```bash
python -m streamlit run app.py
```

### Model Files Not Found

Ensure the trained model files are available in the `models/` directory, or run the training scripts before launching the application.

### Runtime Model Uploads (new)

You can now upload pre-trained model artifacts directly from the Streamlit sidebar at runtime. This allows you to load a BiLSTM or BERT model without restarting the server:

- Go to the Streamlit sidebar and expand **Upload / Load Models (runtime)**.
- Upload `bilstm_final.keras` (or `.h5`), `tokenizer.pkl`, and `label_encoder.pkl` and click **Upload & Load BiLSTM**.
- Upload `bert_model.pt` and click **Upload & Load BERT**.

Uploaded models are saved under the `models/` folder and loaded into the running app immediately.


---

# 📈 Future Improvements

* Voice emotion detection
* Facial emotion recognition
* Student performance analytics
* Personalized learning dashboard
* Multi-language support
* User authentication
* Cloud deployment

---
Video Links For the Project Explanation
Code Demonstration: [https://drive.google.com/file/d/12LUMbhw3RC3hM4wf81Nxz40rjGJp4m64/view?usp=drive_link]


Project Demonstration:[https://drive.google.com/file/d/1vUuiVp43V1kdr24NDTDCOvwIAvwGeF4E/view?usp=drive_link]

# 👩‍💻 Author

**Venkata Sushma Perla**

B.Sc. Computer Science

SRM University-AP

GitHub:
https://github.com/Sushma-1510

---

# 📜 License

This project is developed for academic and educational purposes.

Feel free to modify and extend it for learning and research.
