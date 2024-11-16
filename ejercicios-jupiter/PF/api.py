import os
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Ruta del modelo guardado
BASE_DIR = os.getcwd()
MODEL_FILE = os.path.join(BASE_DIR, 'crime_prediction_model.pkl')

# Cargar modelo entrenado
def load_model():
    print("Cargando el modelo guardado...")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"El archivo {MODEL_FILE} no existe. Por favor, entrena el modelo primero.")
    
    return joblib.load(MODEL_FILE)

# Ruta para predecir Location Description
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del cuerpo de la solicitud
        input_data = request.get_json()
        latitude = input_data['latitude']
        longitude = input_data['longitude']
        primary_type = input_data['primary_type']
        
        # Cargar modelo y codificador
        model, label_encoder = load_model()
        
        # Codificar el tipo primario
        encoded_primary_type = label_encoder.transform([primary_type])[0]
        
        # Crear entrada para el modelo
        X = [[latitude, longitude, encoded_primary_type]]
        
        # Realizar la predicción
        predicted_location = model.predict(X)[0]
        
        return jsonify({'predicted_location_description': predicted_location})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
