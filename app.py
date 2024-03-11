# flask_app.py

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Machine Learning Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the request
        features = request.json['features']
        
        # Make predictions using the loaded model
        prediction = model.predict([features])[0]
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
