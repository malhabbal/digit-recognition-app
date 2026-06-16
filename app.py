# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
import re

# Fallback: use tflite_runtime on Render, full TensorFlow locally
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='model/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_base64):
    # Remove data URL prefix
    img_data = re.sub('^data:image/.+;base64,', '', img_base64)
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale

    # Invert (canvas draws white bg, black strokes -> MNIST expects black bg, white digit)
    img = Image.eval(img, lambda x: 255 - x)

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    img_base64 = data['image']
    try:
        processed_img = preprocess_image(img_base64)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        return jsonify({'digit': digit, 'confidence': round(confidence, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)