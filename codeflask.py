from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # Ganti dengan 'best.pt' kalau lokal

# Ganti dengan link CCTV kamu
STREAM_URL = "https://atcsdishub.pemkomedan.go.id/camera/ISMUDGAJAHMADA.m3u8"

def gen_frames():
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Gagal open stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi helm
        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()

        # Encode sebagai JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        # Kirim sebagai MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h2>Live Stream YOLOv8</h2><img src='/video_feed' width='720'>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
