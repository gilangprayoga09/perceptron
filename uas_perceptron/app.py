from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler from the file
with open('perceptron_model.pkl', 'rb') as model_file:
    model, scaler = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    
    # Translate prediction to weather condition
    weather_conditions = ["Cerah", "Berawan", "Hujan Kecil", "Hujan Petir"]
    weather_condition = weather_conditions[prediction[0]]
    
    return jsonify({'prediction': weather_condition})

if __name__ == '__main__':
    app.run(debug=True)
