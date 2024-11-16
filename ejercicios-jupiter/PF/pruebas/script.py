import os
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib

app = Flask(__name__)

# Ruta absoluta para los archivos
BASE_DIR = os.getcwd()
CSV_FILE = os.path.join(BASE_DIR, 'Crimes_-_2024_20241031.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'location_model.pkl')

# Cargar y procesar datos
def load_and_process_data():
    print("Cargando el archivo CSV...")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"El archivo {CSV_FILE} no existe.")
    
    data = pd.read_csv(CSV_FILE)
    
    # Seleccionar columnas relevantes
    features = ['Latitude', 'Longitude', 'Primary Type']
    target = 'Location Description'
    
    # Eliminar filas con valores nulos en las columnas seleccionadas
    data = data.dropna(subset=features + [target])
    
    # Codificar la columna 'Primary Type' como valores numéricos
    label_encoder = LabelEncoder()
    data['Primary Type'] = label_encoder.fit_transform(data['Primary Type'])
    
    print("Datos procesados correctamente.")
    return data, features, target, label_encoder

# Entrenar modelo
def train_model():
    print("Iniciando el entrenamiento del modelo...")
    data, features, target, label_encoder = load_and_process_data()
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Modelo entrenado correctamente.")
    
    # Guardar el modelo y el codificador en un archivo
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump((model, label_encoder), f)
        print(f"Modelo guardado correctamente en {MODEL_FILE}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
joblib.dump(model, 'crime_prediction_model.pkl')

# Cargar modelo entrenado
def load_model():
    print("Cargando el modelo guardado...")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"El archivo {MODEL_FILE} no existe. Por favor, entrena el modelo primero.")
    
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

# Ruta para entrenar el modelo
@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({'message': 'Modelo entrenado y guardado con éxito.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    # Descomentar esta línea para entrenar automáticamente al iniciar el script
    # train_model()
    app.run(debug=True)
