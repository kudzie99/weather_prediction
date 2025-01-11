# app.py
from flask import Flask, render_template, request, jsonify
import pickle
from datetime import datetime
import requests
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained models correctly
def load_models(filename):
    """Load saved weather prediction models and scaler from a file."""
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    return models['regressor'], models['classifier'], models['scaler']

try:
    # Load models using the correct unpacking
    regressor, classifier, scaler = load_models('C:\\projects\weather\\app\\weather_models.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Get API key from environment variable
API_KEY = os.getenv(API_KEY)
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f'{BASE_URL}weather?q={city}&units=metric&appid={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            'city': data['name'],
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon'],
            'current_temp': round(data['main']['temp'], 2),
            'feels_like': round(data['main']['feels_like'], 2),
            'min_temp': round(data['main']['temp_min'], 2),
            'max_temp': round(data['main']['temp_max'], 2),
            'humidity': round(data['main']['humidity']),
            'pressure': round(data['main']['pressure'], 2),
            'cloud': data['clouds']['all'],
            'timezone': data['timezone'],
            'description': data['weather'][0]['description'].capitalize(),
            'wind_degree': data['wind']['deg'],
            'country': data['sys']['country'],
            'wind_kph': round(data['wind']['speed'] * 3.6, 2),
            'gust_mps': round(data['wind']['gust'], 2) if 'gust' in data['wind'] else 0,
            'visibility_m': data['visibility'],
            'rain': 1 if 'rain' in data else 0
        }
    return None

def prepare_current_weather(weather_data):
    hour = (weather_data['timezone'] % 86400) // 3600

    features = {
        'latitude': weather_data['latitude'],
        'longitude': weather_data['longitude'],
        'wind_kph': weather_data['wind_kph'],
        'wind_degree': weather_data['wind_degree'],
        'pressure_mb': weather_data['pressure'],
        'cloud': weather_data['cloud'],
        'visibility_m': weather_data['visibility_m'],
        'gust_mps': weather_data['gust_mps'],
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24)
    }

    day_part = pd.cut([hour], bins=[-1, 6, 12, 18, 24],
                      labels=['night', 'morning', 'afternoon', 'evening'])[0]
    features.update({
        f'day_part_{part}': 1 if part == day_part else 0
        for part in ['night', 'morning', 'afternoon', 'evening']
    })

    temp = weather_data['current_temp']
    humidity = weather_data['humidity']
    pressure = weather_data['pressure']
    wind = weather_data['wind_kph']

    features.update({
        'temp_squared': temp ** 2,
        'humidity_squared': humidity ** 2,
        'pressure_squared': pressure ** 2,
        'wind_squared': wind ** 2,
        'temp_humidity_interaction': temp * humidity / 100,
        'wind_temp_interaction': wind * temp,
        'pressure_temp_interaction': pressure * temp / 1000
    })

    feature_columns = [
        'latitude', 'longitude', 'wind_kph', 'wind_degree', 'pressure_mb',
        'cloud', 'visibility_m', 'gust_mps', 'hour_sin', 'hour_cos',
        'temp_squared', 'humidity_squared', 'pressure_squared', 'wind_squared',
        'temp_humidity_interaction', 'wind_temp_interaction',
        'pressure_temp_interaction', 'day_part_night', 'day_part_morning',
        'day_part_afternoon', 'day_part_evening'
    ]

    df = pd.DataFrame([features])
    return df[feature_columns]

def predict_weather(weather_data, hours=5):
    predictions = []
    base_features = prepare_current_weather(weather_data)
    current_hour = (weather_data['timezone'] % 86400) // 3600

    for hour in range(1, hours + 1):
        future_hour = (current_hour + hour) % 24
        hour_features = base_features.copy()

        hour_features['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
        hour_features['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)

        day_part = pd.cut([future_hour], bins=[-1, 6, 12, 18, 24],
                         labels=['night', 'morning', 'afternoon', 'evening'])[0]
        for part in ['night', 'morning', 'afternoon', 'evening']:
            hour_features[f'day_part_{part}'] = 1 if part == day_part else 0

        X_scaled = scaler.transform(hour_features)
        reg_pred = regressor.predict(X_scaled)
        rain_prob = classifier.predict_proba(X_scaled)[0][1]

        predictions.append({
            'hour': f'{future_hour:02d}:00',
            'temperature': round(reg_pred[0][0], 1),
            'humidity': round(reg_pred[0][1], 1),
            'rain_probability': round(rain_prob * 100, 1)
        })

    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weather', methods=['POST'])
def weather():
    city = request.form.get('city')
    current_weather = get_current_weather(city)
    
    if current_weather:
        predictions = predict_weather(current_weather)
        return jsonify({
            'status': 'success',
            'current': current_weather,
            'forecast': predictions
        })
    
    return jsonify({
        'status': 'error',
        'message': 'City not found'
    })

if __name__ == '__main__':
    app.run(debug=True)