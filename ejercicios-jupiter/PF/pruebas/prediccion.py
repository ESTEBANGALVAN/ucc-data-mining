import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import numpy as np

# Cargar datos
df = pd.read_csv('Crimes_-_2024_20241031.csv')

# Limpiar datos
df['IUCR'] = pd.to_numeric(df['IUCR'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['IUCR', 'Year'], inplace=True)

# Crear copia de los datos
X = df[['IUCR', 'Primary Type', 'Year']].copy()
y = df['Location Description'].copy()

# Convertir variables categóricas
le_primary = LabelEncoder()
le_location = LabelEncoder()

X['Primary Type'] = le_primary.fit_transform(X['Primary Type'])
y = le_location.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Crear API
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    IUCR = datos['IUCR']
    Primary_Type = datos['Primary Type']
    Year = datos['Year']

    try:
        # Convertir variable categórica
        Primary_Type = le_primary.transform([Primary_Type])[0]
        prediccion = modelo.predict([[IUCR, Primary_Type, Year]])
        return jsonify({'Location Description': le_location.inverse_transform([prediccion[0]])[0]})
    except ValueError as e:
        return jsonify({"error": "Categoría no encontrada"}), 400

if __name__ == '__main__':
    app.run(debug=True)