from ultralytics import YOLO
import cv2
import os
import random

# Load trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Load image
image_path = "0273.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gagal membaca gambar dari path: {image_path}")

# Set confidence threshold
conf_threshold = 0.4

# Predict with confidence threshold
results = model(image, conf=conf_threshold)[0]

# Warna untuk tiap kelas (BGR), akan ditambahkan otomatis jika belum ada
class_colors = {}

# Tampilkan hasil prediksi
for box in results.boxes:
    conf = box.conf[0].item()
    if conf < conf_threshold:
        continue  # skip prediksi yang tidak memenuhi threshold

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    label = model.names[cls_id]

    # Buat warna unik untuk kelas jika belum ada
    if cls_id not in class_colors:
        class_colors[cls_id] = tuple(random.randint(0, 255) for _ in range(3))
    color = class_colors[cls_id]

    # Gambar kotak dan teks label + confidence
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Simpan hasil
output_path = "colored_prediction.jpg"
cv2.imwrite(output_path, image)
print(f"Hasil disimpan ke: {output_path}")

# Tampilkan hasil
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
