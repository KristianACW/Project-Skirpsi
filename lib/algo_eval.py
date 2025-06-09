
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import mediapipe as mp

# === Dataset Warna Foundation ===
color_samples = {
    "Light Ivory 22N Wardah Colorfit Matte Foundation": np.array([233, 194, 163]),
    "Freebies Warm Ivory Wardah Colorfit Matte Foundation": np.array([250, 244, 244]),
    "Pink Fair C11 Wardah Colorfit Matte Foundation": np.array([233, 195, 182]),
    "Warm Ivory 23W Wardah Colorfit Matte Foundation": np.array([230, 185, 146]),
    "Beige 32N Wardah Colorfit Matte Foundation": np.array([216, 165, 138]),
    "Golden Sand (200) INFALLIBLE 24h Fresh Wear": np.array([231, 185, 149]),
    "Ivory (110) ColorStay Full Cover Foundation": np.array([255, 219, 197]),
    "Buff (150) ColorStay Full Cover Foundation": np.array([244, 214, 180]),
}

def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    mask = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1 + strength, mask, -strength, 0)
    return sharpened

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

def extract_neck_color(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = apply_unsharp_mask(image)
    image = apply_clahe(image)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = image.shape
        face_landmarks = results.multi_face_landmarks[0]
        idxs = [152, 377, 400, 378]
        points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in idxs]

        x_min = max(min(p[0] for p in points) - 10, 0)
        x_max = min(max(p[0] for p in points) + 10, w)
        y_min = max(min(p[1] for p in points) - 10, 0)
        y_max = min(max(p[1] for p in points) + 10, h)

        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        avg_color = np.mean(roi, axis=(0, 1))
        return avg_color[::-1]  # RGB

def batch_evaluate(folder_path, save_csv=False):
    success, fail = 0, 0
    results = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, filename)
        print(f"Evaluating: {filename}")
        avg_color = extract_neck_color(path)
        detected = avg_color is not None
        if detected:
            success += 1
        else:
            fail += 1
        results.append({
            "filename": filename,
            "detected": detected
        })

    total = success + fail
    print("\n=== HASIL EVALUASI ===")
    print(f"[MediaPipe] Berhasil: {success}, Gagal: {fail}, Akurasi: {success / total * 100:.2f}%")

    if save_csv:
        with open("evaluasi_mediapipe.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "detected"])
            writer.writeheader()
            writer.writerows(results)
        print(">> CSV tersimpan: evaluasi_mediapipe.csv")

    labels = ['MediaPipe']
    success_vals = [success]
    fail_vals = [fail]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, success_vals, width, label='Berhasil')
    ax.bar(x + width/2, fail_vals, width, label='Gagal')
    ax.set_ylabel('Jumlah Gambar')
    ax.set_title('Hasil Deteksi Leher dengan MediaPipe')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = r"D:\Flutter app\Dart\Flutter Prc\flutter_application_1\flutter_skripsi\Project-Skirpsi\foundationfit-backend\test_images"
    batch_evaluate(folder, save_csv=True)


# === Tambahan Fungsi Evaluasi Algoritma ===

def find_best_match_cosine(input_color):
    best_match = None
    best_score = -1
    for name, color in color_samples.items():
        score = cosine_similarity([input_color], [color])[0][0]
        if score > best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def find_best_match_euclidean(input_color):
    best_match = None
    best_score = float('inf')
    for name, color in color_samples.items():
        score = euclidean(input_color, color)
        if score < best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def find_best_match_manhattan(input_color):
    best_match = None
    best_score = float('inf')
    for name, color in color_samples.items():
        score = cityblock(input_color, color)
        if score < best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def batch_match_algorithm(folder_path):
    cosine_count = euclidean_count = manhattan_count = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, filename)
        avg_color = extract_neck_color(path)
        if avg_color is None:
            continue

        result_cosine, score_cosine = find_best_match_cosine(avg_color)
        result_euclidean, score_euclidean = find_best_match_euclidean(avg_color)
        result_manhattan, score_manhattan = find_best_match_manhattan(avg_color)

        scores = {
            "Cosine": score_cosine,
            "Euclidean": score_euclidean,
            "Manhattan": score_manhattan
        }

        best_algorithm = max(scores.items(), key=lambda x: x[1] if x[0] == "Cosine" else -x[1])[0]

        if best_algorithm == "Cosine":
            cosine_count += 1
        elif best_algorithm == "Euclidean":
            euclidean_count += 1
        else:
            manhattan_count += 1

    print("\n=== FREKUENSI ALGORTIMA TERPILIH ===")
    print(f"Cosine: {cosine_count} kali")
    print(f"Euclidean: {euclidean_count} kali")
    print(f"Manhattan: {manhattan_count} kali")

if __name__ == "__main__":
    folder = r"D:\Flutter app\Dart\Flutter Prc\flutter_application_1\flutter_skripsi\Project-Skirpsi\foundationfit-backend\test_images"
    batch_match_algorithm(folder)


# === Logging CSV Algoritma Terbaik per Gambar ===

def batch_match_algorithm(folder_path, save_csv=False):
    cosine_count = euclidean_count = manhattan_count = 0
    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, filename)
        avg_color = extract_neck_color(path)
        if avg_color is None:
            continue

        result_cosine, score_cosine = find_best_match_cosine(avg_color)
        result_euclidean, score_euclidean = find_best_match_euclidean(avg_color)
        result_manhattan, score_manhattan = find_best_match_manhattan(avg_color)

        scores = {
            "Cosine": score_cosine,
            "Euclidean": score_euclidean,
            "Manhattan": score_manhattan
        }

        best_algorithm = max(scores.items(), key=lambda x: x[1] if x[0] == "Cosine" else -x[1])[0]

        if best_algorithm == "Cosine":
            cosine_count += 1
        elif best_algorithm == "Euclidean":
            euclidean_count += 1
        else:
            manhattan_count += 1

        results.append({
            "filename": filename,
            "cosine_score": score_cosine,
            "euclidean_score": score_euclidean,
            "manhattan_score": score_manhattan,
            "selected_algorithm": best_algorithm
        })

    print("\n=== FREKUENSI ALGORTIMA TERPILIH ===")
    print(f"Cosine: {cosine_count} kali")
    print(f"Euclidean: {euclidean_count} kali")
    print(f"Manhattan: {manhattan_count} kali")

    if save_csv:
        with open("hasil_algoritma_terbaik.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "cosine_score", "euclidean_score", "manhattan_score", "selected_algorithm"])
            writer.writeheader()
            writer.writerows(results)
        print(">> CSV tersimpan: hasil_algoritma_terbaik.csv")

if __name__ == "__main__":
    folder = r"D:\Flutter app\Dart\Flutter Prc\flutter_application_1\flutter_skripsi\Project-Skirpsi\foundationfit-backend\test_images"
    batch_match_algorithm(folder, save_csv=True)
