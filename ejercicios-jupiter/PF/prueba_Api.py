import requests
import json

# URL de la API
url = 'http://127.0.0.1:5000/predict'

# Datos de entrada
data = {
    "latitude": 41.773373505,
    "longitude": -87.582951936,
    "primary_type": "CRIMINAL TRESPASS"
}

# Enviar la solicitud POST
response = requests.post(url, json=data)

# Mostrar la respuesta como texto para depuración
print(response.text)

# Si la respuesta es JSON válido, convertirlo
if response.status_code == 200:
    print(response.json())
else:
    print("Error al obtener la respuesta.")
