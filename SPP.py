import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("Accurate Stock Price Predictor")
st.subheader("Train and forecast prices for any stock")

# ---------------------------- USER INPUTS ---------------------------- #
stock_ticker = st.text_input("Enter Stock Ticker (e.g. AAPL or TATASTEEL.NS):", "").strip().upper()

interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d']
interval = st.selectbox("Select Candle Interval", interval_options, index=8)

use_calendar = st.checkbox("Use calendar (select start and end dates)", value=False)

duration_options = {
    "7d": 7,
    "14d": 14,
    "30d": 30,
    "60d": 60,
    "90d": 90,
    "180d": 180,
    "240d": 240,
    "360d": 360
}
selected_duration = None
start_date = None
end_date = None

today = datetime.date.today()

if use_calendar:
    start_date = st.date_input("Start Date", today - datetime.timedelta(days=180))
    end_date = st.date_input("End Date", today)
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()
else:
    selected_duration = st.selectbox("Select Duration Range", list(duration_options.keys()), index=5)
    end_date = today
    start_date = end_date - datetime.timedelta(days=duration_options[selected_duration])

forecast_days = st.number_input("Days to forecast", min_value=1, max_value=180, value=5, step=1)
train_button = st.button("Train & Predict")

# ---------------------------- DATA FETCHING ---------------------------- #
@st.cache_data(ttl=1800)
def fetch_data(ticker: str, start: datetime.date, end: datetime.date, interval: str, period_for_intraday: str = None) -> pd.DataFrame:
    """
    Fetch data from yfinance respecting interval constraints.
    If period_for_intraday is provided, use period mode for intraday intervals.
    """
    intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']

    if interval in intraday_intervals:
        # must use period for intraday; period_for_intraday should be like '7d'
        period = period_for_intraday or '7d'
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end + datetime.timedelta(days=1), interval=interval, progress=False)
    return df

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ---------------------------- VALIDATION ---------------------------- #
if not stock_ticker:
    st.info("Enter a valid stock ticker to continue.")
    st.stop()

if train_button:
    try:
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        # Enforce Yahoo rules if intraday interval selected
        if interval in intraday_intervals:
            st.warning("Intraday intervals (1m–1h) require `period` mode and support up to ~7 days. Calendar dates will be ignored for intraday requests.")
            # determine period to request: prefer selected_duration if it's <=7d and not using calendar
            if use_calendar:
                period = '7d'
            else:
                # map selected_duration like '14d' to allowed closest period
                allowed = ['1d', '2d', '5d', '7d']
                try:
                    days = duration_options.get(selected_duration, 7)
                except Exception:
                    days = 7
                if days <= 1:
                    period = '1d'
                elif days <= 2:
                    period = '2d'
                elif days <= 5:
                    period = '5d'
                else:
                    period = '7d'
            df = fetch_data(stock_ticker, start_date, end_date, interval, period_for_intraday=period)
        else:
            # daily or higher intervals: use calendar if selected, otherwise duration-derived start/end
            df = fetch_data(stock_ticker, start_date, end_date, interval, period_for_intraday=None)

        if df.empty:
            st.error("No data retrieved for the given parameters.")
            st.stop()

        st.dataframe(df.tail(100))

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close_data = df[[close_col]].dropna()

        if len(close_data) < 120:
            st.warning(f"Insufficient data ({len(close_data)} rows). Need at least 120.")
            st.stop()

        # ---------------------------- PREPROCESSING ---------------------------- #
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_data)

        base_days = 100
        x, y = [], []
        for i in range(base_days, len(scaled)):
            x.append(scaled[i - base_days:i, 0])
            y.append(scaled[i, 0])
        x = np.array(x).reshape(-1, base_days, 1)
        y = np.array(y).reshape(-1, 1)

        # ---------------------------- MODEL TRAINING ---------------------------- #
        uniq = f"{stock_ticker}_{interval}"
        if not use_calendar and selected_duration:
            uniq += f"_{selected_duration}"
        model_path = f"/tmp/model_{uniq}.h5"
        retrain = st.checkbox("Retrain even if cached model exists", value=False)

        if os.path.exists(model_path) and not retrain:
            model = load_model(model_path)
            st.success("Loaded cached model")
        else:
            st.info("Training model...")
            model = build_model((x.shape[1], 1))
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
            ]
            model.fit(x, y, epochs=75, batch_size=32, validation_split=0.1, verbose=1, callbacks=callbacks)
            model.save(model_path)
            st.success("Model trained successfully")

        # ---------------------------- PREDICTIONS ---------------------------- #
        pred_scaled = model.predict(x, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled)
        y_actual = scaler.inverse_transform(y)
        results_df = pd.DataFrame({"Predicted": pred_actual.flatten(), "Actual": y_actual.flatten()})

        st.markdown("### Prediction vs Actual")
        st.line_chart(results_df)

        # ---------------------------- FUTURE FORECAST ---------------------------- #
        st.markdown(f"### {forecast_days}-Day Forecast")
        last_window = list(scaled[-base_days:].flatten())
        future_scaled = []

        for _ in range(int(forecast_days)):
            inp = np.array(last_window[-base_days:]).reshape(1, base_days, 1)
            nxt = model.predict(inp, verbose=0)[0, 0]
            future_scaled.append(nxt)
            last_window.append(nxt)

        future_actual = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1))
        future_df = pd.DataFrame(future_actual, columns=["Forecast Price"])
        st.dataframe(future_df)
        st.line_chart(future_df)

        # ---------------------------- DOWNLOADS ---------------------------- #
        st.download_button(
            label="Download Prediction CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{stock_ticker}_{interval}_predictions.csv",
            mime="text/csv"
        )

        st.download_button(
            label=f"Download {forecast_days}-Day Forecast CSV",
            data=future_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{stock_ticker}_{forecast_days}_day_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
