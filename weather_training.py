import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import requests
import pickle
import json
import os

df = pd.read_csv("GlobalWeatherRepository.csv")

df = df.drop(['last_updated_epoch',
              'last_updated',
              'temperature_fahrenheit',
              'condition_text',
              'wind_mph',
              'wind_direction',
              'pressure_in',
              'feels_like_fahrenheit',
              'visibility_miles',
              'gust_mph',
              'uv_index',
              'air_quality_Carbon_Monoxide',
              'air_quality_Ozone',
              'air_quality_Nitrogen_dioxide',
              'air_quality_Sulphur_dioxide',
              'air_quality_PM2.5',
              'air_quality_PM10',
              'air_quality_us-epa-index',
              'air_quality_gb-defra-index',
              'sunrise',
              'sunset',
              'moonrise',
              'moonset',
              'moon_phase',
              'moon_illumination'], axis=1)

offset_map = {
    'Asia/Kabul': 16200,  # +04:30
    'Europe/Tirane': 3600,  # +01:00
    'Africa/Algiers': 3600,  # +01:00
    'Europe/Andorra': 3600,  # +01:00
    'Africa/Luanda': 3600,  # +01:00
    'America/Antigua': -14400,  # -04:00
    'America/Argentina/Buenos_Aires': -10800,  # -03:00
    'Asia/Yerevan': 14400,  # +04:00
    'Australia/Sydney': 36000,  # +10:00
    'Europe/Vienna': 3600,  # +01:00
    'Asia/Baku': 14400,  # +04:00
    'America/Nassau': -18000,  # -05:00
    'Asia/Bahrain': 10800,  # +03:00
    'Asia/Dhaka': 21600,  # +06:00
    'America/Barbados': -14400,  # -04:00
    'Europe/Minsk': 10800,  # +03:00
    'Europe/Brussels': 3600,  # +01:00
    'America/Belize': -21600,  # -06:00
    'Africa/Porto-Novo': 3600,  # +01:00
    'Asia/Thimphu': 21600,  # +06:00
    'America/La_Paz': -14400,  # -04:00
    'Europe/Sarajevo': 3600,  # +01:00
    'Africa/Gaborone': 7200,  # +02:00
    'America/Manaus': -14400,  # -04:00
    'Asia/Brunei': 28800,  # +08:00
    'Europe/Sofia': 7200,  # +02:00
    'Africa/Ouagadougou': 0,  # +00:00
    'Africa/Bujumbura': 7200,  # +02:00
    'Indian/Antananarivo': 10800,  # +03:00
    'Atlantic/Cape_Verde': -3600,  # -01:00
    'Asia/Phnom_Penh': 25200,  # +07:00
    'Africa/Douala': 3600,  # +01:00
    'America/Toronto': -18000,  # -05:00
    'Africa/Bangui': 3600,  # +01:00
    'America/Santiago': -14400,  # -04:00
    'Asia/Shanghai': 28800,  # +08:00
    'Indian/Comoro': 10800,  # +03:00
    'Africa/Brazzaville': 3600,  # +01:00
    'America/Costa_Rica': -21600,  # -06:00
    'Europe/Zagreb': 3600,  # +01:00
    'America/Havana': -18000,  # -05:00
    'Asia/Famagusta': 7200,  # +02:00
    'Europe/Prague': 3600,  # +01:00
    'Europe/Copenhagen': 3600,  # +01:00
    'Africa/Djibouti': 10800,  # +03:00
    'America/Dominica': -14400,  # -04:00
    'America/Santo_Domingo': -14400,  # -04:00
    'America/Guayaquil': -18000,  # -05:00
    'Africa/Cairo': 7200,  # +02:00
    'America/El_Salvador': -21600,  # -06:00
    'Africa/Malabo': 3600,  # +01:00
    'Africa/Asmara': 10800,  # +03:00
    'Europe/Tallinn': 7200,  # +02:00
    'Africa/Mbabane': 7200,  # +02:00
    'Africa/Addis_Ababa': 10800,  # +03:00
    'Pacific/Fiji': 43200,  # +12:00
    'Europe/Helsinki': 7200,  # +02:00
    'Europe/Paris': 3600,  # +01:00
    'Africa/Libreville': 3600,  # +01:00
    'Africa/Banjul': 0,  # +00:00
    'Asia/Tbilisi': 14400,  # +04:00
    'Europe/Berlin': 3600,  # +01:00
    'Africa/Accra': 0,  # +00:00
    'Europe/Athens': 7200,  # +02:00
    'America/Grenada': -14400,  # -04:00
    'America/Guatemala': -21600,  # -06:00
    'Africa/Conakry': 0,  # +00:00
    'Africa/Bissau': 0,  # +00:00
    'America/Guyana': -14400,  # -04:00
    'America/Port-au-Prince': -18000,  # -05:00
    'Europe/Rome': 3600,  # +01:00
    'America/Tegucigalpa': -21600,  # -06:00
    'Europe/Budapest': 3600,  # +01:00
    'Atlantic/Reykjavik': 0,  # +00:00
    'Asia/Kolkata': 19800,  # +05:30
    'Asia/Jakarta': 25200,  # +07:00
    'Asia/Tehran': 12600,  # +03:30
    'Asia/Baghdad': 10800,  # +03:00
    'Europe/Dublin': 0,  # +00:00
    'Asia/Jerusalem': 7200,  # +02:00
    'America/Jamaica': -18000,  # -05:00
    'Asia/Tokyo': 32400,  # +09:00
    'Asia/Amman': 7200,  # +02:00
    'Asia/Almaty': 21600,  # +06:00
    'Africa/Nairobi': 10800,  # +03:00
    'Pacific/Tarawa': 43200,  # +12:00
    'Asia/Kuwait': 10800,  # +03:00
    'Asia/Bishkek': 21600,  # +06:00
    'Europe/Riga': 7200,  # +02:00
    'Asia/Beirut': 7200,  # +02:00
    'Africa/Johannesburg': 7200,  # +02:00
    'Africa/Monrovia': 0,  # +00:00
    'Asia/Bangkok': 25200,  # +07:00
    'Europe/Vaduz': 3600,  # +01:00
    'Europe/Vilnius': 7200,  # +02:00
    'Europe/Luxembourg': 3600,  # +01:00
    'Africa/Blantyre': 7200,  # +02:00
    'Asia/Kuala_Lumpur': 28800,  # +08:00
    'Indian/Maldives': 18000,  # +05:00
    'Africa/Bamako': 0,  # +00:00
    'Europe/Malta': 3600,  # +01:00
    'Pacific/Majuro': 43200,  # +12:00
    'Africa/Nouakchott': 0,  # +00:00
    'Indian/Mauritius': 14400,  # +04:00
    'America/Mexico_City': -21600,  # -06:00
    'Pacific/Pohnpei': 39600,  # +11:00
    'Europe/Monaco': 3600,  # +01:00
    'Asia/Ulaanbaatar': 28800,  # +08:00
    'Europe/Podgorica': 3600,  # +01:00
    'Africa/Casablanca': 3600,  # +01:00
    'Africa/Maputo': 7200,  # +02:00
    'Asia/Yangon': 23400,  # +06:30
    'Africa/Windhoek': 7200,  # +02:00
    'Europe/Istanbul': 7200,  # +02:00
    'Asia/Kathmandu': 20700,  # +05:45
    'Europe/Amsterdam': 3600,  # +01:00
    'Pacific/Auckland': 43200,  # +12:00
    'America/Managua': -21600,  # -06:00
    'Africa/Niamey': 3600,  # +01:00
    'Africa/Lagos': 3600,  # +01:00
    'Asia/Pyongyang': 32400,  # +09:00
    'Europe/Skopje': 3600,  # +01:00
    'Europe/Oslo': 3600,  # +01:00
    'Asia/Muscat': 14400,  # +04:00
    'Asia/Karachi': 18000,  # +05:00
    'Pacific/Palau': 32400,  # +09:00
    'America/Panama': -18000,  # -05:00
    'Pacific/Port_Moresby': 36000,  # +10:00
    'America/Lima': -18000,  # -05:00
    'Asia/Manila': 28800,  # +08:00
    'Europe/Warsaw': 3600,  # +01:00
    'Europe/Lisbon': 0,  # +00:00
    'Asia/Qatar': 10800,  # +03:00
    'Europe/Bucharest': 7200,  # +02:00
    'Europe/Moscow': 10800,  # +03:00
    'Africa/Kigali': 7200,  # +02:00
    'America/St_Kitts': -14400,  # -04:00
    'America/St_Lucia': -14400,  # -04:00
    'America/St_Vincent': -14400,  # -04:00
    'Pacific/Apia': 46800,  # +13:00
    'Asia/Riyadh': 10800,  # +03:00
    'Africa/Dakar': 0,  # +00:00
    'Europe/Belgrade': 3600,  # +01:00
    'Indian/Mahe': 14400,  # +04:00
    'Africa/Freetown': 0,  # +00:00
    'Asia/Singapore': 28800,  # +08:00
    'Europe/Bratislava': 3600,  # +01:00
    'Europe/Ljubljana': 3600,  # +01:00
    'Pacific/Guadalcanal': 39600,  # +11:00
    'Africa/Mogadishu': 10800,  # +03:00
    'Asia/Seoul': 32400,  # +09:00
    'Africa/Khartoum': 7200,  # +02:00
    'Europe/Madrid': 3600,  # +01:00
    'Asia/Colombo': 19800,  # +05:30
    'America/Paramaribo': -10800,  # -03:00
    'Europe/Stockholm': 3600,  # +01:00
    'Europe/Zurich': 3600,  # +01:00
    'Asia/Damascus': 7200,  # +02:00
    'Asia/Dushanbe': 18000,  # +05:00
    'Africa/Dar_es_Salaam': 10800,  # +03:00
    'Asia/Dili': 32400,  # +09:00
    'Pacific/Tongatapu': 46800,  # +13:00
    'America/Port_of_Spain': -14400,  # -04:00
    'Africa/Tunis': 3600,  # +01:00
    'Asia/Ashgabat': 18000,  # +05:00
    'Pacific/Funafuti': 43200,  # +12:00
    'Africa/Kampala': 10800,  # +03:00
    'Europe/Kiev': 7200,  # +02:00
    'Asia/Dubai': 14400,  # +04:00
    'Europe/London': 0,  # +00:00
    'America/Los_Angeles': -28800,  # -08:00
    'America/Montevideo': -10800,  # -03:00
    'Asia/Tashkent': 18000,  # +05:00
    'Pacific/Efate': 39600,  # +11:00
    'America/Caracas': -16200,  # -04:30
    'Asia/Aden': 10800,  # +03:00
    'Africa/Lusaka': 7200,  # +02:00
    'Africa/Harare': 7200,  # +02:00
    'America/Asuncion': -14400,  # -04:00
    'America/Argentina/Cordoba': -10800,  # -03:00
    'America/Sao_Paulo': -10800,  # -03:00
    'America/Bogota': -18000,  # -05:00
    'America/Denver': -25200,  # -07:00
    'America/Chicago': -21600,  # -06:00
    'America/New_York': -18000,  # -05:00
    'Africa/Tripoli': 7200,  # +02:00
    'Asia/Ho_Chi_Minh': 25200,  # +07:00
    'Australia/Melbourne': 36000,  # +10:00
    'Asia/Kuching': 28800,  # +08:00
    'America/Hermosillo': -25200  # -07:00
}

df['timezone'] = df['timezone'].map(offset_map)

df = df.drop(['country', 'location_name', 'precip_in'], axis=1)

df['visibility_km'] = df['visibility_km']*1000
df['gust_kph'] = (df['gust_kph']/3.6).round(2)

df.rename(columns={'visibility_km': 'visibility_m',
                   'gust_kph': 'gust_mps'}, inplace=True)

df['rain'] = df['precip_mm'].apply(lambda x: 1 if x > 0.0 else 0)
df = df.drop('precip_mm', axis=1)


# Access the variable
API_KEY = os.getenv(API_KEY)
BASE_URL= 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f'{BASE_URL}weather?q={city}&units=metric&appid={API_KEY}'
    response = requests.get(url)

    # Check if the request was successful
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
            'description': data['weather'][0]['description'],
            'wind_degree': data['wind']['deg'],
            'country': data['sys']['country'],
            'wind_kph': round(data['wind']['speed'], 2),
            'gust_mps': round(data['wind']['gust'], 2) if 'gust' in data['wind'] else 0, # Handle missing gust data
            'visibility_m': data['visibility'],
            'rain': 1 if 'rain' in data else 0

        }
    else:
        print(f"Error fetching weather data for {city}. Status code: {response.status_code}")
        print("Please check the city name or try again later.")
        return None # or handle the error appropriately

def prepare_features(df):
    """Prepare enhanced features for model training/prediction"""
    # Create time-based features
    df['hour'] = (df['timezone'] % 86400) // 3600
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Add more time-based features
    df['day_part'] = pd.cut(df['hour'],
                           bins=[-1, 6, 12, 18, 24],
                           labels=['night', 'morning', 'afternoon', 'evening'])
    df = pd.get_dummies(df, columns=['day_part'])

    # Create interaction features
    df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity'] / 100
    df['wind_temp_interaction'] = df['wind_kph'] * df['temperature_celsius']
    df['pressure_temp_interaction'] = df['pressure_mb'] * df['temperature_celsius'] / 1000

    # Add squared terms for important features
    df['temp_squared'] = df['temperature_celsius'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    df['pressure_squared'] = df['pressure_mb'] ** 2
    df['wind_squared'] = df['wind_kph'] ** 2

    # Feature columns in fixed order
    feature_columns = [
        'latitude',
        'longitude',
        'wind_kph',
        'wind_degree',
        'pressure_mb',
        'cloud',
        'visibility_m',
        'gust_mps',
        'hour_sin',
        'hour_cos',
        'temp_squared',
        'humidity_squared',
        'pressure_squared',
        'wind_squared',
        'temp_humidity_interaction',
        'wind_temp_interaction',
        'pressure_temp_interaction',
        'day_part_night',
        'day_part_morning',
        'day_part_afternoon',
        'day_part_evening'
    ]

    return df[feature_columns]

def train_models(X, y):
    """Train enhanced models with optimized parameters"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optimize regressor parameters
    reg_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42
    }

    base_regressor = RandomForestRegressor(**reg_params)
    regressor = MultiOutputRegressor(base_regressor)
    regressor.fit(X_scaled, y[['temperature_celsius', 'humidity']])

    # Optimize classifier parameters
    clf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    }

    rain_classifier = RandomForestClassifier(**clf_params)
    rain_classifier.fit(X_scaled, y['rain'])

    return regressor, rain_classifier, scaler

def prepare_current_weather(weather_data):
    """Prepare current weather data with enhanced features"""
    hour = (weather_data['timezone'] % 86400) // 3600

    # Create base features
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

    # Add time-based features
    day_part = pd.cut([hour], bins=[-1, 6, 12, 18, 24],
                      labels=['night', 'morning', 'afternoon', 'evening'])[0]
    features.update({
        f'day_part_{part}': 1 if part == day_part else 0
        for part in ['night', 'morning', 'afternoon', 'evening']
    })

    # Add squared and interaction terms
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

    # Create DataFrame with fixed column order
    feature_columns = [
        'latitude',
        'longitude',
        'wind_kph',
        'wind_degree',
        'pressure_mb',
        'cloud',
        'visibility_m',
        'gust_mps',
        'hour_sin',
        'hour_cos',
        'temp_squared',
        'humidity_squared',
        'pressure_squared',
        'wind_squared',
        'temp_humidity_interaction',
        'wind_temp_interaction',
        'pressure_temp_interaction',
        'day_part_night',
        'day_part_morning',
        'day_part_afternoon',
        'day_part_evening'
    ]

    df = pd.DataFrame([features])
    return df[feature_columns]

def split_target_variables(df):
    """
    Split the dataset into features and target variables for weather prediction.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing weather data

    Returns:
    tuple: (X, y) where X contains feature data and y contains target variables
    """
    # First prepare the features using the existing prepare_features function
    X = prepare_features(df)

    # Create target DataFrame with temperature, humidity and rain
    y = pd.DataFrame({
        'temperature_celsius': df['temperature_celsius'],
        'humidity': df['humidity'],
        'rain': df['rain']  # Assuming this is a binary column (0 or 1)
    })

    return X, y

def predict_weather(weather_data, regressor, rain_classifier, scaler, hours=5):
    """Make enhanced predictions with uncertainty estimates"""
    predictions = []
    base_features = prepare_current_weather(weather_data)
    current_hour = (weather_data['timezone'] % 86400) // 3600

    for hour in range(1, hours + 1):
        future_hour = (current_hour + hour) % 24
        hour_features = base_features.copy()

        # Update time features
        hour_features['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
        hour_features['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)

        # Update day part features
        day_part = pd.cut([future_hour], bins=[-1, 6, 12, 18, 24],
                         labels=['night', 'morning', 'afternoon', 'evening'])[0]
        for part in ['night', 'morning', 'afternoon', 'evening']:
            hour_features[f'day_part_{part}'] = 1 if part == day_part else 0

        # Scale features
        X_scaled = scaler.transform(hour_features)

        # Make direct prediction for temperature and humidity
        reg_pred = regressor.predict(X_scaled)

        # Get individual predictions for uncertainty estimation
        temp_predictions = []
        humidity_predictions = []

        # Access individual estimators for each target
        for est_temp, est_humid in zip(regressor.estimators_[0].estimators_,
                                     regressor.estimators_[1].estimators_):
            temp_pred = est_temp.predict(X_scaled)
            humidity_pred = est_humid.predict(X_scaled)
            temp_predictions.append(temp_pred[0])
            humidity_predictions.append(humidity_pred[0])

        temp_predictions = np.array(temp_predictions)
        humidity_predictions = np.array(humidity_predictions)

        # Calculate uncertainties
        temp_std = np.std(temp_predictions)
        humidity_std = np.std(humidity_predictions)

        # Get rain probability
        rain_prob = rain_classifier.predict_proba(X_scaled)[0][1]

        predictions.append({
            'hour': future_hour,
            'temperature_celsius': round(reg_pred[0][0], 2),
            'temperature_uncertainty': round(temp_std, 2),
            'humidity': round(reg_pred[0][1], 2),
            'humidity_uncertainty': round(humidity_std, 2),
            'rain_probability': round(rain_prob * 100, 2)
        })

    return predictions

def evaluate_models(X, y, regressor, rain_classifier, scaler):
    """Evaluate model performance"""
    X_scaled = scaler.transform(X)

    # Evaluate regression models
    reg_pred = regressor.predict(X_scaled)
    temp_mse = mean_squared_error(y['temperature_celsius'], reg_pred[:, 0])
    humidity_mse = mean_squared_error(y['humidity'], reg_pred[:, 1])

    # Evaluate classifier
    rain_pred = rain_classifier.predict(X_scaled)
    rain_accuracy = accuracy_score(y['rain'], rain_pred)

    return {
        'temperature_rmse': np.sqrt(temp_mse),
        'humidity_rmse': np.sqrt(humidity_mse),
        'rain_accuracy': rain_accuracy
    }

import pickle

def save_models(filename, regressor, classifier, scaler):
    """
    Save trained weather prediction models and scaler to a file.

    Parameters:
    filename (str): Path where models should be saved
    regressor (MultiOutputRegressor): Trained regressor for temperature and humidity
    classifier (RandomForestClassifier): Trained classifier for rain prediction
    scaler (StandardScaler): Fitted scaler for feature normalization
    """
    models = {
        'regressor': regressor,
        'classifier': classifier,
        'scaler': scaler
    }

    with open(filename, 'wb') as f:
        pickle.dump(models, f)

def load_models(filename):
    """
    Load saved weather prediction models and scaler from a file.

    Parameters:
    filename (str): Path to saved models file

    Returns:
    tuple: (regressor, classifier, scaler)
    """
    with open(filename, 'rb') as f:
        models = pickle.load(f)

    return models['regressor'], models['classifier'], models['scaler']

# Example usage
def main():
    # Split data for training and evaluation
    X, y = split_target_variables(df)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Train models
    regressor, rain_classifier, scaler = train_models(X_train, y_train)

    # Evaluate models
    metrics = evaluate_models(X_val, y_val, regressor, rain_classifier, scaler)
    print("\nModel Performance Metrics:")
    print(f"Temperature RMSE: {metrics['temperature_rmse']:.2f}°C")
    print(f"Humidity RMSE: {metrics['humidity_rmse']:.2f}%")
    print(f"Rain Prediction Accuracy: {metrics['rain_accuracy']:.2f}")

    # Save models
    save_models('weather_models.pkl', regressor, rain_classifier, scaler)

    # Make predictions
    city = input("Enter city name: ")
    current_weather = get_current_weather(city)
    if current_weather:
        predictions = predict_weather(current_weather, regressor, rain_classifier, scaler)

        print("\nWeather Predictions for the next 5 hours:")
        for pred in predictions:
            print(f"\nHour {pred['hour']}:")
            print(f"Temperature: {pred['temperature_celsius']}°C ± {pred['temperature_uncertainty']}°C")
            print(f"Humidity: {pred['humidity']}% ± {pred['humidity_uncertainty']}%")
            print(f"Rain Probability: {pred['rain_probability']}%")

if __name__ == "__main__":
    main()

