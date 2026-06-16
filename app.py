# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import re

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model/model.h5')

def preprocess_image(img_base64):
    """
    Decode base64 image, convert to grayscale, resize to 28x28,
    invert if necessary (white digit on black background), normalize,
    and return a numpy array of shape (1, 28, 28, 1).
    """
    # Remove data URL prefix (e.g., "data:image/png;base64,")
    img_data = re.sub('^data:image/.+;base64,', '', img_base64)
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
    
    # Invert if background is white and digit is black (MNIST expects black background)
    # Usually canvas draws white bkg and black strokes, so we invert:
    img = Image.eval(img, lambda x: 255 - x)
    
    # Resize to 28x28 using anti-aliasing
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
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
        predictions = model.predict(processed_img, verbose=0)
        digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        return jsonify({'digit': digit, 'confidence': round(confidence, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)