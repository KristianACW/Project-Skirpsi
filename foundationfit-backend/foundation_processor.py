import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import mediapipe as mp

# Dataset warna foundation
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

def extract_neck_color(image_path):
    """Ekstraksi rata-rata warna dari area leher pada wajah dalam gambar."""
    image = cv2.imread(image_path)
    if image is None:
        return None

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

        # Landmark area leher bagian bawah
        idxs = [152, 377, 400, 378]
        points = [
            (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
            for i in idxs
        ]

        x_min = max(min(p[0] for p in points) - 10, 0)
        x_max = min(max(p[0] for p in points) + 10, w)
        y_min = max(min(p[1] for p in points) - 10, 0)
        y_max = min(max(p[1] for p in points) + 10, h)

        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        avg_color = np.mean(roi, axis=(0, 1))  # BGR
        return avg_color[::-1]  # Kembalikan dalam format RGB

def find_best_match_cosine(input_color):
    """Cari kecocokan warna terbaik menggunakan cosine similarity."""
    best_match = None
    best_score = -1
    for name, color in color_samples.items():
        score = cosine_similarity([input_color], [color])[0][0]
        if score > best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def find_best_match_euclidean(input_color):
    """Cari kecocokan warna terbaik menggunakan jarak Euclidean."""
    best_match = None
    best_score = float('inf')
    for name, color in color_samples.items():
        score = euclidean(input_color, color)
        if score < best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def find_best_match_manhattan(input_color):
    """Cari kecocokan warna terbaik menggunakan jarak Manhattan."""
    best_match = None
    best_score = float('inf')
    for name, color in color_samples.items():
        score = cityblock(input_color, color)
        if score < best_score:
            best_score = score
            best_match = name
    return best_match, best_score

def to_hex(rgb):
    """Konversi RGB ke kode warna HEX."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_best_foundation_match(image_path):
    """Fungsi utama untuk memilih foundation terbaik berdasarkan warna leher."""
    avg_color = extract_neck_color(image_path)
    if avg_color is None:
        return None, None, None

    # Hitung kecocokan untuk semua algoritma
    result_cosine, score_cosine = find_best_match_cosine(avg_color)
    result_euclidean, score_euclidean = find_best_match_euclidean(avg_color)
    result_manhattan, score_manhattan = find_best_match_manhattan(avg_color)

    # Konversi ke HEX
    hex_cosine = to_hex(color_samples[result_cosine])
    hex_euclidean = to_hex(color_samples[result_euclidean])
    hex_manhattan = to_hex(color_samples[result_manhattan])

    # Tentukan algoritma terbaik
    scores = {
        "Cosine": score_cosine,        # Semakin tinggi lebih baik
        "Euclidean": score_euclidean,  # Semakin rendah lebih baik
        "Manhattan": score_manhattan   # Semakin rendah lebih baik
    }

    # Logika pemilihan: Cosine maksimum, lainnya minimum
    best_algorithm = max(scores.items(), key=lambda x: x[1] if x[0] == "Cosine" else -x[1])[0]

    # Tentukan hasil akhir berdasarkan algoritma terbaik
    if best_algorithm == "Cosine":
        best_brand = result_cosine
        best_hex = hex_cosine
    elif best_algorithm == "Euclidean":
        best_brand = result_euclidean
        best_hex = hex_euclidean
    else:
        best_brand = result_manhattan
        best_hex = hex_manhattan

    print(f"Selected algorithm: {best_algorithm}")
    print(f"Selected brand: {best_brand}")
    print(f"Selected hex: {best_hex}")
    print("Score Cosine:", score_cosine, "| Result:", result_cosine)
    print("Score Euclidean:", score_euclidean, "| Result:", result_euclidean)
    print("Score Manhattan:", score_manhattan, "| Result:", result_manhattan)

    return best_algorithm, best_brand, best_hex
