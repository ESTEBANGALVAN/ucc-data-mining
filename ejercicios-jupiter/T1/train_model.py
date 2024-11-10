import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
# Paso 1: Cargar los datos
data = pd.read_csv('Crimes_-_2024_20241031.csv')
# Paso 2: Preprocesamiento de datos
# Eliminar columnas innecesarias o que no sean útiles para la predicción
data = data.drop(columns=['Case Number', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location Description'])
# Codificar las variables categóricas con LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
   le = LabelEncoder()
   data[column] = le.fit_transform(data[column])
   label_encoders[column] = le
# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(columns=['Primary Type'])
y = data['Primary Type']
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Paso 3: Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Paso 4: Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Paso 5: Guardar el modelo entrenado
joblib.dump(model, 'crime_prediction_model.pkl')