# =========================================================
# APPLE STOCK PRICE FORECASTING WEB APPLICATION
# FINAL DEPLOYMENT USING STREAMLIT + LSTM
# =========================================================

# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# Streamlit is used to build the web application UI
import streamlit as st

# Pandas is used for data handling and dataframe operations
import pandas as pd

# NumPy is used for numerical computations
import numpy as np

# Matplotlib is used for plotting graphs
import matplotlib.pyplot as plt

import plotly.graph_objects as go
# Load trained LSTM model
from tensorflow.keras.models import load_model

# Load saved scaler object
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

# Configure Streamlit page settings
st.set_page_config(

    # Browser tab title
    page_title="Apple Stock Forecast",

    # Wide screen layout
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; color:#222222;'>🍎 Apple Stock Price Forecasting Dashboard</h1>",
    unsafe_allow_html=True
)

# =========================================================
# CUSTOM UI DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #ff2b2b;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙ Dashboard Controls")

future_days = 30

show_dataset = st.sidebar.checkbox(
    "Show Dataset",
    value=True
)

history_days = st.sidebar.slider(
    "Historical Days to Display",
    30,
    365,
    100
)


# =========================================================
# LOAD DATASET
# =========================================================

# Read Apple stock dataset
df = pd.read_csv("P668 DATASET.csv")


# =========================================================
# DATA PREPROCESSING
# =========================================================

# Convert Date column into datetime format
# This allows proper time-series operations
df['Date'] = pd.to_datetime(df['Date'])

# Set Date column as dataframe index
# Makes plotting and forecasting easier
df.set_index('Date', inplace=True)

# Moving averages
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()


# =========================================================
# KPI METRICS
# =========================================================

latest_price = round(df['Close'].iloc[-1], 2)

highest_price = round(df['Close'].max(), 2)

lowest_price = round(df['Close'].min(), 2)

col1, col2, col3 = st.columns(3)

col1.metric("📌 Latest Price", f"${latest_price}")

col2.metric("📈 Highest Price", f"${highest_price}")

col3.metric("📉 Lowest Price", f"${lowest_price}")

# =========================================================
# DISPLAY DATASET
# =========================================================

# Show dataset preview
if show_dataset:

    st.subheader("📄 Historical Dataset")

    st.dataframe(df.head(10))
    st.info(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# =========================================================
# LOAD TRAINED LSTM MODEL
# =========================================================

# Load saved trained model
model = load_model("lstm_model.keras")
# =========================================================
# LOAD SAVED SCALER
# =========================================================

# Load scaler used during training
# Important because prediction input must be scaled
scaler = joblib.load("scaler.pkl")


# =========================================================
# SCALE CLOSE PRICES
# =========================================================

# Scale Close prices between 0 and 1
# LSTM performs better on normalized data
scaled_close = scaler.transform(df[['Close']])


# =========================================================
# DISPLAY HISTORICAL GRAPH
# =========================================================

# =========================================================
# INTERACTIVE HISTORICAL GRAPH
# =========================================================

st.markdown("## 📊 Historical Stock Analysis")

# Create Plotly figure
fig = go.Figure()

# =====================================================
# CLOSE PRICE LINE
# =====================================================

fig.add_trace(
    go.Scatter(
        x=df.index[-history_days:],
        y=df['Close'].tail(history_days),
        mode='lines',
        name='Close Price'
    )
)

# MA20
fig.add_trace(
    go.Scatter(
        x=df.index[-history_days:],
        y=df['MA20'].tail(history_days),
        mode='lines',
        name='MA20'
    )
)

# MA50
fig.add_trace(
    go.Scatter(
        x=df.index[-history_days:],
        y=df['MA50'].tail(history_days),
        mode='lines',
        name='MA50'
    )
)

# =====================================================
# GRAPH LAYOUT
# =====================================================

fig.update_layout(
    template="plotly_dark",
    title="Historical Apple Stock Prices",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    hovermode="x unified"
)

# =====================================================
# DISPLAY GRAPH
# =====================================================

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# FORECAST BUTTON
# =========================================================

if st.button("Generate Forecast"):

    with st.spinner("Generating Forecast..."):

        # Last 60 days sequence
        last_sequence = scaled_close[-60:]

        # Store predictions
        future_predictions = []

        # Copy sequence
        current_sequence = last_sequence.copy()

        # Forecast loop
        for i in range(future_days):

            # Reshape for LSTM
            X_future = current_sequence.reshape(
                1,
                60,
                1
            )

            # Predict next value
            next_pred = model.predict(
                X_future,
                verbose=0
            )

            # Store prediction
            future_predictions.append(
                next_pred[0,0]
            )

            # Update sequence
            current_sequence = np.append(
                current_sequence[1:],
                next_pred
            )

        # Convert predictions back to original scale
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1,1)
        )

        st.success("Forecast Generated Successfully!")

    # =====================================================
    # CREATE FUTURE DATES
    # =====================================================

    future_dates = pd.date_range(
        start=df.index[-1],
        periods=future_days + 1,
        freq='B'
    )[1:]

    # =====================================================
    # CREATE FORECAST DATAFRAME
    # =====================================================

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': future_predictions.flatten()
    })

    forecast_df['Predicted Close Price'] = forecast_df['Predicted Close Price'].round(2)

   # =====================================================
   # FORECAST KPI CARDS
   # =====================================================

    st.markdown("## 📌 Forecast Summary")

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "📈 Max Forecast",
        f"${round(forecast_df['Predicted Close Price'].max(),2)}"
    )

    f2.metric(
        "📉 Min Forecast",
        f"${round(forecast_df['Predicted Close Price'].min(),2)}"
    )

    f3.metric(
        "📊 Average Forecast",
        f"${round(forecast_df['Predicted Close Price'].mean(),2)}"
    )

    trend = forecast_df['Predicted Close Price'].iloc[-1] - forecast_df['Predicted Close Price'].iloc[0]

    f4.metric(
        "📈 Forecast Trend",
        f"${round(trend,2)}"
    )    

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="apple_forecast.csv",
        mime="text/csv"
    )

    # =====================================================
    # DISPLAY FORECAST TABLE
    # =====================================================

    st.subheader("📅 Forecasted Prices")

    st.dataframe(forecast_df)

    # =====================================================
    # FORECAST VISUALIZATION
    # =====================================================

    st.markdown("## 🔮 Future Price Prediction")

    fig2 = go.Figure()

    # Historical data
    fig2.add_trace(
        go.Scatter(
            x=df.index[-history_days:],
            y=df['Close'].iloc[-history_days:],
            mode='lines',
            name='Historical Data'
        )
    )

    # Forecast data
    fig2.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_predictions.flatten(),
            mode='lines',
            name='Forecast',
            line=dict(color='cyan', width=3)
        )
    )

    fig2.update_layout(
        template="plotly_dark",
        title="Apple Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================

st.write("""
-----------------------------------------------------------
Developed using:
- Streamlit
- TensorFlow/Keras
- LSTM Deep Learning Model
-----------------------------------------------------------
""")