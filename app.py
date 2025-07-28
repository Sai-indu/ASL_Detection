from flask import Flask, render_template, request, jsonify
from model import predict_image
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((64, 64))  # adjust based on your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = predict_image(img_array)
    predicted_class = int(np.argmax(predictions))

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
