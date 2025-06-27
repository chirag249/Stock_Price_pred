import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import datetime

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title('📈 Accurate Stock Price Predictor')
st.subheader('Train & Predict prices for any stock dynamically')

# User Inputs
stock_ticker = st.text_input('Enter Stock Ticker (e.g., AAPL or TATASTEEL.NS):', '')
start_date = st.date_input('Start Date', datetime.date(2024, 1, 1))
end_date = st.date_input('End Date', datetime.date(2025, 6, 25))
forecast_days = st.number_input("Select number of days to forecast:", min_value=1, max_value=180, value=5, step=1)

if stock_ticker and start_date < end_date:
    if st.button("🚀 Train & Predict"):
        try:
            # Fetch data
            df = yf.download(stock_ticker, start=start_date, end=end_date)
            if df.empty or len(df) < 300:
                st.warning("Not enough data. Try a longer date range.")
                st.stop()

            df.reset_index(inplace=True)
            st.dataframe(df.tail(3000))

            # Preprocessing
            close_data = df[['Close']]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_data)

            base_days = 100
            x, y = [], []
            for i in range(base_days, len(scaled_data)):
                x.append(scaled_data[i - base_days:i, 0])
                y.append(scaled_data[i, 0])
            x = np.array(x).reshape(-1, base_days, 1)
            y = np.array(y)

            # Build model
            st.info("Training model... Please wait ⏳")
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(100, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x, y, epochs=75, batch_size=32, verbose=1)
            st.success("✅ Model trained successfully")

            # Predict on training data
            pred = model.predict(x)
            pred_actual = scaler.inverse_transform(pred)
            y_actual = scaler.inverse_transform(y.reshape(-1, 1))
            results_df = pd.DataFrame({
                'Predicted Price': pred_actual.flatten(),
                'Actual Price': y_actual.flatten()
            })
            st.markdown("### 📊 Prediction vs Actual")
            st.line_chart(results_df)

            # Download predictions
            pred_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=pred_csv,
                file_name=f"{stock_ticker}_predictions.csv",
                mime="text/csv"
            )

            # Forecast future prices
            st.markdown(f"### 🔮 Future {forecast_days}-Day Forecast")
            test_input = list(scaled_data[-base_days:].flatten())
            future_predictions = []

            for _ in range(forecast_days):
                input_batch = np.array(test_input[-base_days:]).reshape(1, base_days, 1)
                next_pred = model.predict(input_batch, verbose=0)[0][0]
                future_predictions.append(next_pred)
                test_input.append(next_pred)

            future_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_df = pd.DataFrame(future_actual, columns=["Future Price"])

            # Debug and validate forecast
            st.write("Forecast values (scaled back):", future_df)

            if not future_df.empty and future_df["Future Price"].notnull().all():
                st.line_chart(future_df)
            else:
                st.warning("⚠️ Forecast data is invalid or empty.")

            # Download future forecast
            future_csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {forecast_days}-Day Forecast",
                data=future_csv,
                file_name=f"{stock_ticker}_{forecast_days}_day_forecast.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("Enter a valid stock ticker and date range to begin.")
