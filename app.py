"""
Customer Classification API - Flask Application
Production-ready API with Neural Network, Gradient Boosting, and Random Forest models.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Model paths
MODEL_NN_PATH = 'model_nn.h5'
MODEL_GB_PATH = 'model_gb.pkl'
MODEL_RF_PATH = 'model_rf.pkl'
SCALER_PATH = 'scaler.pkl'

# Global model storage
models = {}
scaler = None

def load_artifacts():
    global models, scaler
    try:
        if os.path.exists(MODEL_NN_PATH):
            models['nn'] = load_model(MODEL_NN_PATH)
            print("✓ Neural Network loaded")
        if os.path.exists(MODEL_GB_PATH):
            with open(MODEL_GB_PATH, 'rb') as f:
                models['gb'] = pickle.load(f)
            print("✓ Gradient Boosting loaded")
        if os.path.exists(MODEL_RF_PATH):
            with open(MODEL_RF_PATH, 'rb') as f:
                models['rf'] = pickle.load(f)
            print("✓ Random Forest loaded")
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded")
        return scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Load on startup
scaler = load_artifacts()

# Segment labels
SEGMENTS = {0: 'Low Value', 1: 'Mid Value', 2: 'High Value'}
SEGMENT_COLORS = {0: '#ef4444', 1: '#3b82f6', 2: '#10b981'}
SEGMENT_DESCRIPTIONS = {
    0: 'At-risk customer with low engagement. Consider re-engagement campaigns and win-back offers.',
    1: 'Regular customer with moderate value. Target for upselling opportunities and loyalty programs.',
    2: 'VIP customer with high value. Provide premium treatment, personalized offers, and priority support.'
}

MODEL_NAMES = {
    'nn': 'Neural Network',
    'gb': 'Gradient Boosting',
    'rf': 'Random Forest'
}


@app.route('/')
def home():
    """Render the web UI"""
    available_models = list(models.keys())
    return render_template('index.html', models=available_models)


@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    if models and scaler is not None:
        return jsonify({
            'status': 'healthy',
            'models_loaded': list(models.keys()),
            'scaler_loaded': True
        }), 200
    return jsonify({'status': 'unhealthy', 'models_loaded': False}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict customer segment from RFM features.
    
    Expected JSON body:
    {
        "recency": 30,
        "frequency": 5,
        "monetary": 500.0,
        "model": "nn"  // Optional: "nn", "gb", "rf"
    }
    """
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        recency = float(data.get('recency', 0))
        frequency = float(data.get('frequency', 0))
        monetary = float(data.get('monetary', 0))
        model_type = data.get('model', 'nn')  # Default to neural network
        
        # Validate inputs
        if recency < 0 or frequency < 0 or monetary < 0:
            return jsonify({'error': 'All values must be non-negative'}), 400
        
        if model_type not in models:
            return jsonify({'error': f'Model "{model_type}" not available. Choose from: {list(models.keys())}'}), 400
        
        # Preprocess: Log transform
        features = np.array([[recency, frequency, monetary]])
        features_log = np.log(features + 1)
        features_scaled = scaler.transform(features_log)
        
        # Get predictions from all models for comparison
        all_predictions = {}
        
        for name, model in models.items():
            if name == 'nn':
                probs = model.predict(features_scaled, verbose=0)[0]
                pred = int(np.argmax(probs))
                all_predictions[name] = {
                    'segment': pred,
                    'segment_name': SEGMENTS[pred],
                    'probabilities': {
                        'low': round(float(probs[0]) * 100, 2),
                        'mid': round(float(probs[1]) * 100, 2),
                        'high': round(float(probs[2]) * 100, 2)
                    }
                }
            else:
                pred = int(model.predict(features_scaled)[0])
                probs = model.predict_proba(features_scaled)[0]
                all_predictions[name] = {
                    'segment': pred,
                    'segment_name': SEGMENTS[pred],
                    'probabilities': {
                        'low': round(float(probs[0]) * 100, 2),
                        'mid': round(float(probs[1]) * 100, 2),
                        'high': round(float(probs[2]) * 100, 2)
                    }
                }
        
        # Get primary prediction from selected model
        primary = all_predictions[model_type]
        probs = primary['probabilities']
        prediction = primary['segment']
        confidence = max(probs['low'], probs['mid'], probs['high'])
        
        result = {
            'model_used': MODEL_NAMES[model_type],
            'segment': prediction,
            'segment_name': SEGMENTS[prediction],
            'confidence': confidence,
            'description': SEGMENT_DESCRIPTIONS[prediction],
            'color': SEGMENT_COLORS[prediction],
            'probabilities': probs,
            'input': {
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            },
            'all_models': all_predictions
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
