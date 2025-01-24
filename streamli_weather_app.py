import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request
import io

# Page config
st.set_page_config(
    page_title="Weather Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced metric styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    div[data-testid="metric-container"] > div {
        width: 100%;
    }
    div[data-testid="metric-container"] label {
        color: #0f0f0f;
        font-weight: 600;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #2c3e50;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Load models from GitHub
@st.cache_resource
def load_github_models(url):
    """Load saved weather prediction models and scaler from GitHub."""
    try:
        response = urllib.request.urlopen(url)
        models = pickle.load(response)
        return models['regressor'], models['classifier'], models['scaler']
    except Exception as e:
        st.error(f"Error loading models from GitHub: {e}")
        return None, None, None

# Get the model URL from Streamlit secrets
GITHUB_MODEL_URL = st.secrets["github_model_url"]
regressor, classifier, scaler = load_github_models(GITHUB_MODEL_URL)

if not all([regressor, classifier, scaler]):
    st.error("Failed to load models. Please check your configuration.")
    st.stop()

# Get API key from Streamlit secrets
API_KEY = st.secrets["weather_api_key"]
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    """Fetch current weather data from OpenWeatherMap API."""
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
    """Prepare weather data for model prediction."""
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
    """Predict weather for the next few hours."""
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

def create_forecast_plot(predictions):
    """Create a plotly figure for forecast visualization."""
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Rain Probability (%)'))
    
    hours = [p['hour'] for p in predictions]
    temps = [p['temperature'] for p in predictions]
    humidity = [p['humidity'] for p in predictions]
    rain_prob = [p['rain_probability'] for p in predictions]
    
    fig.add_trace(go.Scatter(x=hours, y=temps, mode='lines+markers', name='Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=humidity, mode='lines+markers', name='Humidity'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=rain_prob, mode='lines+markers', name='Rain Probability'), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def main():
    st.title("Weather Forecast Application")
    
    # Search bar
    city = st.text_input("Enter city name", "")
    
    if city:
        current_weather = get_current_weather(city)
        
        if current_weather:
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Current Weather")
                
                # Adding spacing
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Current weather metrics in two rows
                row1_cols = st.columns(3)
                with row1_cols[0]:
                    st.metric(
                        "Temperature",
                        f"{current_weather['current_temp']}°C",
                        delta=None
                    )
                with row1_cols[1]:
                    st.metric(
                        "Feels Like",
                        f"{current_weather['feels_like']}°C",
                        delta=None
                    )
                with row1_cols[2]:
                    st.metric(
                        "Humidity",
                        f"{current_weather['humidity']}%",
                        delta=None
                    )
                
                # Adding spacing between rows
                st.markdown("<br>", unsafe_allow_html=True)
                
                row2_cols = st.columns(3)
                with row2_cols[0]:
                    st.metric(
                        "Wind Speed",
                        f"{current_weather['wind_kph']} km/h",
                        delta=None
                    )
                with row2_cols[1]:
                    st.metric(
                        "Pressure",
                        f"{current_weather['pressure']} hPa",
                        delta=None
                    )
                with row2_cols[2]:
                    st.metric(
                        "Visibility",
                        f"{current_weather['visibility_m']/1000:.1f} km",
                        delta=None
                    )
                
                # Description
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"Weather Description: {current_weather['description']}")
                
            with col2:
                # Map
                m = folium.Map(location=[current_weather['latitude'], current_weather['longitude']], zoom_start=10)
                folium.Marker(
                    [current_weather['latitude'], current_weather['longitude']],
                    popup=f"{city}: {current_weather['current_temp']}°C"
                ).add_to(m)
                st_folium(m, height=300)
            
            # Forecast section
            st.subheader("Weather Forecast")
            predictions = predict_weather(current_weather)
            
            # Create and display forecast plot
            fig = create_forecast_plot(predictions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast data in a table
            df_forecast = pd.DataFrame(predictions)
            st.dataframe(df_forecast, hide_index=True)
            
        else:
            st.error("City not found. Please check the spelling and try again.")

if __name__ == "__main__":
    main()
