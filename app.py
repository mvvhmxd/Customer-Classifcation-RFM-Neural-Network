"""
Customer Classification API - Flask Application
Production-ready API with Neural Network, Gradient Boosting, and Random Forest models.
Features: Single prediction, Batch CSV, Customer Insights, PDF Export
"""

import os
import io
import csv
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
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

# Reference statistics for percentile calculation (from training data)
RFM_STATS = {
    'recency': {'mean': 92, 'std': 100, 'min': 1, 'max': 374, 'median': 51},
    'frequency': {'mean': 4.3, 'std': 7.6, 'min': 1, 'max': 210, 'median': 2},
    'monetary': {'mean': 1898, 'std': 8219, 'min': 3, 'max': 280206, 'median': 648}
}


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


scaler = load_artifacts()

SEGMENTS = {0: 'Low Value', 1: 'Mid Value', 2: 'High Value'}
SEGMENT_COLORS = {0: '#ef4444', 1: '#eab308', 2: '#22c55e'}
SEGMENT_DESCRIPTIONS = {
    0: 'At-risk customer with low engagement. Consider re-engagement campaigns and win-back offers.',
    1: 'Regular customer with moderate value. Target for upselling opportunities and loyalty programs.',
    2: 'VIP customer with high value. Provide premium treatment, personalized offers, and priority support.'
}
MODEL_NAMES = {'nn': 'Neural Network', 'gb': 'Gradient Boosting', 'rf': 'Random Forest'}


def calculate_percentile(value, stat_key):
    """Calculate approximate percentile based on reference statistics."""
    stats = RFM_STATS[stat_key]
    # Using z-score approximation for percentile
    z = (value - stats['mean']) / stats['std']
    # Clamp to reasonable range
    percentile = 50 + z * 30
    return max(1, min(99, int(percentile)))


def get_insights(recency, frequency, monetary):
    """Generate customer insights with percentile rankings."""
    r_percentile = 100 - calculate_percentile(recency, 'recency')  # Lower recency is better
    f_percentile = calculate_percentile(frequency, 'frequency')
    m_percentile = calculate_percentile(monetary, 'monetary')
    
    avg_percentile = (r_percentile + f_percentile + m_percentile) / 3
    
    # Generate grades
    def grade(p):
        if p >= 80: return 'A'
        if p >= 60: return 'B'
        if p >= 40: return 'C'
        if p >= 20: return 'D'
        return 'F'
    
    return {
        'recency': {
            'value': recency,
            'percentile': r_percentile,
            'grade': grade(r_percentile),
            'comparison': 'days since last purchase',
            'benchmark': f"Avg: {RFM_STATS['recency']['mean']:.0f} days"
        },
        'frequency': {
            'value': frequency,
            'percentile': f_percentile,
            'grade': grade(f_percentile),
            'comparison': 'total transactions',
            'benchmark': f"Avg: {RFM_STATS['frequency']['mean']:.1f} orders"
        },
        'monetary': {
            'value': monetary,
            'percentile': m_percentile,
            'grade': grade(m_percentile),
            'comparison': 'total spending',
            'benchmark': f"Avg: £{RFM_STATS['monetary']['mean']:.0f}"
        },
        'overall_percentile': int(avg_percentile),
        'overall_grade': grade(avg_percentile)
    }


def predict_single(recency, frequency, monetary, model_type='nn'):
    """Make a single prediction."""
    features = np.array([[recency, frequency, monetary]])
    features_log = np.log(features + 1)
    features_scaled = scaler.transform(features_log)
    
    all_predictions = {}
    for name, model in models.items():
        if name == 'nn':
            probs = model.predict(features_scaled, verbose=0)[0]
        else:
            probs = model.predict_proba(features_scaled)[0]
        
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
    
    primary = all_predictions[model_type]
    probs = primary['probabilities']
    prediction = primary['segment']
    confidence = max(probs['low'], probs['mid'], probs['high'])
    
    return {
        'model_used': MODEL_NAMES[model_type],
        'segment': prediction,
        'segment_name': SEGMENTS[prediction],
        'confidence': confidence,
        'description': SEGMENT_DESCRIPTIONS[prediction],
        'color': SEGMENT_COLORS[prediction],
        'probabilities': probs,
        'input': {'recency': recency, 'frequency': frequency, 'monetary': monetary},
        'all_models': all_predictions,
        'insights': get_insights(recency, frequency, monetary)
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    if models and scaler is not None:
        return jsonify({'status': 'healthy', 'models_loaded': list(models.keys())}), 200
    return jsonify({'status': 'unhealthy'}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Single customer prediction."""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        recency = float(data.get('recency', 0))
        frequency = float(data.get('frequency', 0))
        monetary = float(data.get('monetary', 0))
        model_type = data.get('model', 'nn')
        
        if recency < 0 or frequency < 0 or monetary < 0:
            return jsonify({'error': 'All values must be non-negative'}), 400
        
        if model_type not in models:
            return jsonify({'error': f'Model not available'}), 400
        
        result = predict_single(recency, frequency, monetary, model_type)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        model_type = request.form.get('model', 'nn')
        
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF-8"))
        reader = csv.DictReader(stream)
        
        results = []
        for i, row in enumerate(reader):
            try:
                recency = float(row.get('recency', row.get('Recency', 0)))
                frequency = float(row.get('frequency', row.get('Frequency', 0)))
                monetary = float(row.get('monetary', row.get('Monetary', 0)))
                customer_id = row.get('customer_id', row.get('CustomerID', f'Row_{i+1}'))
                
                pred = predict_single(recency, frequency, monetary, model_type)
                results.append({
                    'customer_id': customer_id,
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary,
                    'segment': pred['segment_name'],
                    'confidence': pred['confidence'],
                    'insights': pred['insights']
                })
            except Exception as e:
                results.append({
                    'customer_id': row.get('customer_id', f'Row_{i+1}'),
                    'error': str(e)
                })
        
        # Summary stats
        segments = [r['segment'] for r in results if 'segment' in r]
        summary = {
            'total': len(results),
            'successful': len(segments),
            'low_value': segments.count('Low Value'),
            'mid_value': segments.count('Mid Value'),
            'high_value': segments.count('High Value')
        }
        
        return jsonify({'results': results, 'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export-csv', methods=['POST'])
def export_csv():
    """Export batch results as CSV."""
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'Confidence', 'Overall Grade'])
        
        for r in results:
            if 'error' not in r:
                writer.writerow([
                    r['customer_id'],
                    r['recency'],
                    r['frequency'],
                    r['monetary'],
                    r['segment'],
                    f"{r['confidence']}%",
                    r.get('insights', {}).get('overall_grade', '-')
                ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='customer_classification_results.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
