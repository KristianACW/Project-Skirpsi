from flask import Flask, request
from flask_cors import CORS
import os
from foundation_processor import get_best_foundation_match

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    algorithm, brand, hex_color = get_best_foundation_match(path)

    if None in (algorithm, brand, hex_color):
        return "Failed to process image or detect neck color", 400

    # Return format sesuai kebutuhan: Cosine|Brand|#hex
    return f"{algorithm};{brand};{hex_color}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
