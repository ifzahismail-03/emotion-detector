# Emotion Detector

A real-time facial expression detection web application built using Python, OpenCV, PyTorch, Hugging Face Transformers, and Flask. The application uses a webcam to capture live video, detects faces using OpenCV's Haar Cascade classifier, and passes the detected face region to a pretrained facial-expression classification model. The model predicts one of seven expressions — Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise — along with a confidence score. The Flask backend handles the camera stream and prediction data, while the web interface displays the live camera feed, detected face, predicted expression, and confidence percentage.

## Technologies Used

Python, OpenCV, PyTorch, Hugging Face Transformers, Flask, NumPy, Pillow, HTML, CSS, and JavaScript.

## How It Works

The webcam captures the video using OpenCV. Each frame is processed to detect faces using a Haar Cascade classifier. When a face is detected, the face region is cropped and converted into the required format before being passed to the pretrained classification model. The model returns the predicted facial expression and confidence score. Flask provides the backend for the application and sends the prediction data to the web interface, where the result is updated in real time.

## Expressions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Project Structure

```text
Emotion_Detector/
├── templates/
│   └── index.html
├── static/
├── model/
├── emotion_model.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

Installation
git clone https://github.com/ifzahismail-03/emotion-detector.git
cd emotion-detector
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run
python main.py

Open http://127.0.0.1:5000 in a web browser and allow camera access when prompted.

Note

The application classifies visible facial expressions from camera images. The predicted expression should not be considered a definitive measurement of a person's actual emotional state.

Author

Ifzah Ismail
GitHub: https://github.com/ifzahismail-03
