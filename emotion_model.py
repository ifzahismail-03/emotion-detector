import torch
from transformers import pipeline

# Load a pretrained facial emotion recognition model
emotion_classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def predict_emotion(face_image):
    """
    Takes a face image and returns the predicted emotion
    and confidence.
    """

    results = emotion_classifier(face_image)

    best_result = results[0]

    emotion = best_result["label"]
    confidence = best_result["score"]

    return emotion, confidence