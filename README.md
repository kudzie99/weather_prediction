# Weather Prediction App

Welcome to the Weather Prediction App! This app predicts temperature, humidity, and the possibility of rain for the next five hours based on current weather conditions fetched from the OpenWeatherMap API. It uses machine learning techniques trained on historical weather data to provide accurate and real-time predictions.

---

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Dataset and Model](#dataset-and-model)
7. [API Integration](#api-integration)
8. [Screenshots](#screenshots)
9. [Contributing](#contributing)
10. [Future Enhancements](#future-enhancements)

---

## Features

- **Real-time Weather Data**: Fetches current weather data from OpenWeatherMap API.
- **Hourly Predictions**: Predicts temperature, humidity, and the possibility of rain for the next five hours.
- **Categorical Rain Prediction**: Outputs whether it will rain (Yes/No).
- **User-friendly Interface**: Built with Flask and LeafletJS, providing an intuitive and responsive design.
- **Customizable API Integration**: Easily switch between free and paid OpenWeatherMap API keys.
- **Scalable Model**: Can adapt to additional weather parameters or extended prediction durations.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Flask, Javascript, Requests
- **API**: OpenWeatherMap API
- **Version Control**: Git
- **IDE**: Visual Studio Code or any Python-compatible IDE

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- OpenWeatherMap API key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/weather-prediction-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd weather-prediction-app
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your OpenWeatherMap API key to the `.env` file:
   ```
   API_KEY=your_api_key_here
   ```
5. Run the app:
   ```bash
   python app.py
   ```

---

## Usage

1. Launch the app by running `app.py`.
2. Enter your location
3. View current weather details fetched from the API.
4. Check predictions for the next five hours, including temperature, humidity, and rain possibility.

---

## How It Works

1. **Data Fetching**:
   - The app retrieves current weather data using the OpenWeatherMap API.
2. **Prediction**:
   - Uses a pre-trained machine learning model to predict weather conditions based on the input.
3. **User Interface**:
   - Displays results in a clean and organized format.

---

## Dataset and Model

- **Dataset**:
  - Historical weather data was collected from Kaggle.
  - Features include temperature, humidity, pressure, wind speed, longitude, latitude, and rain status.
- **Model**:
  - Built using Scikit-learn.
  - Trained on 45,000 rows of high quality data.
  - Evaluated using [mean_squared_error, and accuracy_score].

---

## API Integration

- The app uses the OpenWeatherMap API for real-time data.
- Configure the API key in the `.env` file.
- API documentation: [OpenWeatherMap API Docs](https://openweathermap.org/api).

---

## Screenshots

![Screenshot weather](https://github.com/user-attachments/assets/ef7ee825-f059-432c-ab0f-14956c68d19a)

![Screenshot weather1](https://github.com/user-attachments/assets/6ec280b5-64fe-4b49-a840-e8be2b444e19)

---

## Contributing

I welcome contributions! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

---

## Future Enhancements

- Add support for extended forecasts (e.g., next 24 hours, next 7 days).
- Incorporate additional weather parameters like UV index or visibility.
- Enhance model accuracy with larger datasets and deep learning techniques.
- Integrate notification alerts for severe weather conditions.

---

## Contact

#### Email:
kudzaikaremb@gmail.com
#### Github:
@kudzie99
