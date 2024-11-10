from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import numpy as np

# Paso 1: Cargar el modelo entrenado
model = joblib.load('crime_prediction_model.pkl')

# Cargar el conjunto de datos original para obtener las columnas necesarias
data = pd.read_csv('Crimes_-_2024_20241031.csv')

# Eliminar columnas innecesarias o que no sean útiles para la predicción
data = data.drop(columns=['Case Number', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location'])

# Separar las características (X) y la etiqueta (y)
X = data.drop(columns=['Primary Type'])

# Identificar las columnas categóricas y numéricas
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

# Definir el preprocesador para codificación de columnas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Ajustar el preprocesador con los datos originales
preprocessor.fit(X)

# Paso 2: Definir los modelos de datos para la entrada y salida de la API
class CrimePredictionInput(BaseModel):
    Date: str
    Block: str
    IUCR: str
    Location: str
    Arrest: bool
    Domestic: bool
    Beat: int
    Ward: int

class CrimePredictionOutput(BaseModel):
    PRIMARY_DESCRIPTION: str

# Paso 3: Inicializar la aplicación FastAPI
app = FastAPI()

# Paso 4: Definir el endpoint para hacer predicciones de crimen
@app.post('/predict/')
async def predict_crime(data: CrimePredictionInput):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Asegurarse de que Arrest y Domestic sean booleanos
        input_data['Arrest'] = input_data['Arrest'].astype(bool)
        input_data['Domestic'] = input_data['Domestic'].astype(bool)

        # Convertir Date al formato correcto
        input_data['Date'] = pd.to_datetime(input_data['Date'], format='%m/%d/%Y %I:%M:%S %p')

        # Asegurarse de que todas las columnas necesarias estén presentes
        required_columns = set(X.columns)
        missing_columns = required_columns - set(input_data.columns)
        for column in missing_columns:
            input_data[column] = 0

        # Verificar y manejar valores nulos
        input_data.fillna(0, inplace=True)

        # Preprocesar los datos de entrada
        input_data_transformed = preprocessor.transform(input_data)

        # Verificar y manejar valores nulos en los datos transformados
        input_data_transformed = np.nan_to_num(input_data_transformed)

        # Hacer la predicción
        prediction = model.predict(input_data_transformed)

        # Obtener la descripción primaria del crimen predicho
        primary_description = prediction[0]
        return CrimePredictionOutput(PRIMARY_DESCRIPTION=primary_description)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
