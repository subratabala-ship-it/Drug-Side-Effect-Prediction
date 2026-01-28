from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    drug_name = data.get('drug_name')
    age_group = data.get('age_group')
    disease = data.get('disease')

    # Validation
    if not drug_name or not disease:
        return jsonify({'status': 'error', 'message': 'Please fill all fields'}), 400

    # Mock Prediction Logic (replicating your Streamlit demo)
    # In a real app, you would load your ML model here.
    result = {
        'status': 'success',
        'side_effects': [
            'Nausea (Moderate Risk)',
            'Headache (Low Risk)',
            'Dizziness (Low Risk)'
        ],
        'warning': 'Elderly patients may have higher sensitivity.' if 'Elderly' in age_group else ''
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)