from flask import Flask, render_template, Response, jsonify
import cv2

from emotion_model import predict_emotion


app = Flask(__name__)

latest_emotion = "Detecting..."
latest_confidence = 0


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def generate_frames():

    global latest_emotion, latest_confidence

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, width, height) in faces:

            face = frame[
                y:y + height,
                x:x + width
            ]

            try:

                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                emotion, confidence = predict_emotion(face_rgb)

                latest_emotion = emotion
                latest_confidence = confidence * 100

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                text = f"{emotion}: {confidence * 100:.1f}%"

                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            except Exception as error:

                print("Emotion prediction error:", error)

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/emotion")
def emotion():

    return jsonify({
        "emotion": latest_emotion,
        "confidence": latest_confidence
    })


if __name__ == "__main__":

    print("================================")
    print("   EMOTION DETECTOR WEBSITE")
    print("================================")
    print("Open this in your browser:")
    print("http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )