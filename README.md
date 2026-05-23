# Apple Stock Price Prediction using LSTM

## Overview

This project is a Machine Learning and Deep Learning-based Apple Stock Price Prediction System developed using multiple forecasting models. The system analyzes historical Apple stock data and predicts future stock prices using advanced time-series forecasting techniques. The final deployed model uses LSTM for accurate 30-day stock price forecasting.

## Features
- Historical stock price analysis
- Interactive Streamlit web application
- Moving average and volatility analysis
- Multiple model comparison
- 30-day future stock forecasting
- Forecast CSV download option
- Interactive Plotly visualizations

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Matplotli
- Seaborn

## Machine Learning Models Used
| Model         | MAE   | RMSE  | Verdict           |
| ------------- | ----- | ----- | ----------------- |
| ARIMA         | 25.41 | 34.06 | Basic Performance |
| SARIMA        | 25.39 | 34.04 | Improved          |
| SARIMAX       | 6.90  | 9.46  | Good              |
| Random Forest | 15.43 | 26.04 | Moderate          |
| XGBoost       | 17.03 | 28.26 | Moderate          |
| LSTM          | 3.97  | 5.18  | Selected          |
| GRU           | 5.18  | 6.02  | Good              |

## Final Model Selection
LSTM was selected as the final model because it achieved the lowest MAE and RMSE values while effectively capturing nonlinear and sequential stock market patterns.

## Project Workflow
- Data Loading and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Statistical Modeling
- Machine Learning Modeling
- Deep Learning Modeling
- Model Evaluation and Comparison
- Streamlit Deployment

## Deployment
The project was deployed using Streamlit to create an interactive dashboard where users can:
- View historical stock analysis
- Generate future stock forecasts
- Download forecast results
- Analyze stock trends visually

## How to Run the Project

### 1. Clone the Repository
https://github.com/GayatriWarude/apple-stock-price-prediction.git

### 2. Open Project Folder
cd apple-stock-price-prediction

### 3. Install Required Libraries
pip install -r requirements.txt

### 4. Run the Streamlit Application
streamlit run app.py

## Future Improvements
- Add real-time stock API integration
- Improve prediction accuracy
- Deploy on cloud platforms
- Add multi-stock forecasting support

## Author
Gayatri Warude
