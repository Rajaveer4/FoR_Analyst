import os
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('lstm_audio_deepfake_model.h5')

def extract_mfcc_features(file_path, sample_rate=16000, n_mfcc=13, max_pad_len=100):
    try:
        audio_data, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.T[np.newaxis, ...]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    features = extract_mfcc_features(filepath)
    if features is None:
        return jsonify({'error': 'MFCC extraction failed'}), 500

    prediction = model.predict(features)[0][0]
    result = 'Real' if prediction >= 0.5 else 'Fake'
    return jsonify({'prediction': result, 'confidence': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
