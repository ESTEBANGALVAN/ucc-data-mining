import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Ruta absoluta para los archivos
BASE_DIR = os.getcwd()
CSV_FILE = os.path.join(BASE_DIR, 'Crimes_-_2024_20241031.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'crime_prediction_model.pkl')

# Cargar y procesar datos
def load_and_process_data():
    print("Cargando el archivo CSV...")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"El archivo {CSV_FILE} no existe.")
    
    try:
        # Cargar solo un fragmento de los datos para evitar problemas de memoria
        data = pd.read_csv(CSV_FILE, nrows=30000)  # Limitar filas para pruebas
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo CSV: {e}")
    
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
    try:
        data, features, target, label_encoder = load_and_process_data()
    except Exception as e:
        raise RuntimeError(f"Error al procesar los datos: {e}")
    
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
        raise RuntimeError(f"Error al guardar el modelo: {e}")

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
