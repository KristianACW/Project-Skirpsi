import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import mediapipe as mp

# === Dataset Warna Foundation (disingkat untuk contoh) ===
color_samples = {
    # Dataset Wardah
    "Light Ivory 22N Wardah Colorfit Matte Foundation": np.array([233, 194, 163]),
    "Freebies Warm Ivory Wardah Colorfit Matte Foundation": np.array([250, 244, 244]),
    "Pink Fair C11 Wardah Colorfit Matte Foundation": np.array([233, 195, 182]),
    "Warm Ivory 23W Wardah Colorfit Matte Foundation": np.array([230, 185, 146]),
    "Beige 32N Wardah Colorfit Matte Foundation": np.array([216, 165, 138]),
    "Olive Beige 33W Wardah Colorfit Matte Foundation": np.array([222, 157, 101]),
    "Golden Sand 43W Wardah Colorfit Matte Foundation": np.array([228, 152, 94]),
    "Neutral Sand 42N Wardah Colorfit Matte Foundation": np.array([224, 170, 134]),
    "Almond 52N Wardah Colorfit Matte Foundation": np.array([190, 120, 95]),

    # Dataset Loreal
    "Ivory (020) INFALLIBLE 24h Fresh Wear": np.array([249, 212, 185]),
    "Vanilla (120) INFALLIBLE 24h Fresh Wear": np.array([233, 190, 155]),
    "Natural Rose (125) INFALLIBLE 24h Fresh Wear": np.array([232, 192, 157]),
    "True Beige (130) INFALLIBLE 24h Fresh Wear": np.array([230, 199, 170]),
    "Golden Beige (140) INFALLIBLE 24h Fresh Wear": np.array([222, 178, 141]),
    "Radiant Beige (150) INFALLIBLE 24h Fresh Wear": np.array([215, 169, 135]),
    "Golden Sand (200) INFALLIBLE 24h Fresh Wear": np.array([231, 185, 149]),
    "Radiant Sand (250) INFALLIBLE 24h Fresh Wear": np.array([218, 167, 124]),

    # Dataset Revlon
    "Ivory (110) ColorStay Full Cover Foundation": np.array([255, 219, 197]),
    "Buff (150) ColorStay Full Cover Foundation": np.array([244, 214, 180]),
    "Natural Ochre (175) ColorStay Full Cover Foundation": np.array([248, 206, 168]),
    "Nude (200) ColorStay Full Cover Foundation": np.array([244, 205, 174]),
    "Sand Beige (210) ColorStay Full Cover Foundation": np.array([236, 205, 176]),
    "Natural Beige (220) ColorStay Full Cover Foundation": np.array([236, 194, 156]),
    "Medium Beige (240) ColorStay Full Cover Foundation": np.array([229, 187, 145]),
    "Warm Golden (310) ColorStay Full Cover Foundation": np.array([233, 184, 143]),
    "True Beige (320) ColorStay Full Cover Foundation": np.array([228, 183, 142]),
    "Natural Tan (330) ColorStay Full Cover Foundation": np.array([222, 159, 116]),
    "Early Tan (390) ColorStay Full Cover Foundation": np.array([220, 165, 124]),
    "Caramel (425) ColorStay Full Cover Foundation": np.array([194, 138, 91]),
    "Toast (410) ColorStay Full Cover Foundation": np.array([209, 149, 95]),
    "Almond (405) ColorStay Full Cover Foundation": np.array([196, 132, 88]),
    "Mahogany (420) ColorStay Full Cover Foundation": np.array([148, 88, 54]),
}

# === Fungsi Pendukung ===
def extract_neck_color_haar(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    neck_y_start = y + h
    neck_y_end = neck_y_start + int(h * 0.3)
    neck_y_end = min(neck_y_end, image.shape[0])
    roi = image[neck_y_start:neck_y_end, x:x + w]
    if roi.size == 0:
        return None
    avg_color = np.mean(roi, axis=(0, 1))
    return avg_color[::-1]  # RGB

def extract_neck_color_mediapipe(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0]
        idxs = [152, 377, 400, 378]
        points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in idxs]
        x_min = max(min(p[0] for p in points) - 10, 0)
        x_max = min(max(p[0] for p in points) + 10, w)
        y_min = max(min(p[1] for p in points) - 10, 0)
        y_max = min(max(p[1] for p in points) + 10, h)
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        avg_color = np.mean(roi, axis=(0, 1))
        return avg_color[::-1]  # RGB

# === Fungsi Batch Evaluasi ===
def batch_evaluate(folder_path, save_csv=False):
    mediapipe_success, mediapipe_fail = 0, 0
    haar_success, haar_fail = 0, 0
    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, filename)
        print(f"Evaluating: {filename}")
        mp_color = extract_neck_color_mediapipe(path)
        haar_color = extract_neck_color_haar(path)
        mp_ok = mp_color is not None
        haar_ok = haar_color is not None
        if mp_ok: mediapipe_success += 1
        else: mediapipe_fail += 1
        if haar_ok: haar_success += 1
        else: haar_fail += 1
        results.append({
            "filename": filename,
            "mediapipe_detected": mp_ok,
            "haar_detected": haar_ok
        })

    total = mediapipe_success + mediapipe_fail
    print("\n=== HASIL EVALUASI ===")
    print(f"[MediaPipe] Berhasil: {mediapipe_success}, Gagal: {mediapipe_fail}, Akurasi: {mediapipe_success / total * 100:.2f}%")
    print(f"[Haar Cascade] Berhasil: {haar_success}, Gagal: {haar_fail}, Akurasi: {haar_success / total * 100:.2f}%")

    if save_csv:
        with open("evaluasi_deteksi.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "mediapipe_detected", "haar_detected"])
            writer.writeheader()
            writer.writerows(results)
        print(">> CSV tersimpan: evaluasi_deteksi.csv")

    # Plot hasil
    labels = ['MediaPipe', 'Haar Cascade']
    success = [mediapipe_success, haar_success]
    fail = [mediapipe_fail, haar_fail]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, success, width, label='Berhasil')
    rects2 = ax.bar(x + width/2, fail, width, label='Gagal')

    ax.set_ylabel('Jumlah Gambar')
    ax.set_title('Perbandingan Deteksi Leher: MediaPipe vs Haar Cascade')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

# === Jalankan ===
if __name__ == "__main__":
    folder = r"D:\Flutter app\Dart\Flutter Prc\flutter_application_1\flutter_skripsi\Project-Skirpsi\foundationfit-backend\test_images"
    batch_evaluate(folder, save_csv=True)
