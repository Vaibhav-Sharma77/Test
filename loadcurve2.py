import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import pickle
from datetime import datetime, timedelta

# Load the data with caching using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('delhi_electricity_demand (1).csv')

# Load models for predictions based on selected region
def load_models(region):
    models = {}
    try:
        models['random_forest'] = pickle.load(open(f"{region}_rf_model.pkl", 'rb'))
        models['xgboost'] = pickle.load(open(f"{region}_xgb_model.pkl", 'rb'))
        models['lstm'] = pickle.load(open(f"{region}_lstm_model.pkl", 'rb'))
        return models
    except Exception as e:
        st.error(f"Error loading models for {region}: {e}")
        return None

# Predict load curve using the models
def predict_load_curve(models, weather_data):
    predictions = {}
    reshaped_weather_data = np.array(weather_data).reshape(1, 1, -1)  # Reshaping for LSTM input
    for name, model in models.items():
        if model:
            if name == 'lstm':
                prediction = model.predict(reshaped_weather_data)
            else:
                prediction = model.predict([weather_data])
            predictions[name] = prediction[0]
        else:
            predictions[name] = None
    return predictions

# Main Streamlit application
def main():
    # Load the demand data
    demand_data = load_data()

    # Set the title of the Streamlit app
    st.title("Delhi Electricity Load Curve Prediction")

    # Inject custom CSS to change the color of selected regions
    st.markdown(
        """
        <style>
        .selected-region {
            color: blue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create date input for user to select the desired date
    future_date = st.date_input("Select a Future Date")

    # Add a multiselect box for region selection
    regions = ["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES", "All Regions"]
    selected_regions = st.multiselect("Select Regions", options=regions, default=["DELHI"])

    # Convert date into required format
    future_date_str = future_date.strftime('%Y-%m-%d')

    # Prepare to store all predictions
    all_predictions = {}

    for region in selected_regions:
        if region == "All Regions":
            # If "All Regions" is selected, process all individual regions
            for individual_region in regions[:-1]:  # Exclude "All Regions"
                # Load models for the selected region
                models = load_models(individual_region)
                if models:
                    # Prepare weather data for each hour (dummy weather data for simplicity)
                    for hour in range(24):
                        previous_demand = demand_data[individual_region].iloc[-1]
                        weather_data = [20, 60, 10, previous_demand, hour, future_date.weekday(), future_date.month]  # Dummy weather data
                        predictions = predict_load_curve(models, weather_data)
                        all_predictions.setdefault(individual_region, {}).update({hour: predictions})
        else:
            # Load models for the selected individual region
            models = load_models(region)
            if models:
                # Prepare weather data for prediction at this hour
                for hour in range(24):
                    previous_demand = demand_data[region].iloc[-1]
                    weather_data = [20, 60, 10, previous_demand, hour, future_date.weekday(), future_date.month]  # Dummy weather data
                    predictions = predict_load_curve(models, weather_data)
                    all_predictions.setdefault(region, {}).update({hour: predictions})

    # Display prediction results for each selected region
    st.subheader("Predicted Load Curve")
    for region, predictions in all_predictions.items():
        st.write(f"**{region}**:")
        for hour, preds in predictions.items():
            st.write(f"Hour {hour}:")
            for model_name, prediction in preds.items():
                # Check if prediction is an array or scalar and format accordingly
                if isinstance(prediction, (np.ndarray, list)):
                    prediction_value = prediction[0]  # Take the first value if it's an array
                else:
                    prediction_value = prediction  # Use the scalar directly

                st.write(f"{model_name.capitalize()} Prediction: {prediction_value:.2f} MW")

    # Plot the predicted load curves for all regions
    plt.figure(figsize=(10, 6))
    for region, predictions in all_predictions.items():
        # Collecting predictions for plotting
        plt.plot(range(24), [preds['random_forest'][0] if isinstance(preds['random_forest'], (np.ndarray, list)) else preds['random_forest'] for hour, preds in predictions.items()], marker='o', label=region)
    
    plt.title("Comparative Predicted Load Curves")
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Load (MW)")
    plt.xticks(range(24), rotation=45)
    plt.grid(True)
    plt.legend()  # Add legend for regions
    st.pyplot(plt)  # Display the comparative plot in the Streamlit app

    # Calculate load metrics for the report
    report_data = []
    for region, predictions in all_predictions.items():
        loads = [preds['random_forest'][0] if isinstance(preds['random_forest'], (np.ndarray, list)) else preds['random_forest'] for hour, preds in predictions.items()]
        peak_load = max(loads)
        peak_load_time = loads.index(peak_load)
        min_load = min(loads)
        min_load_time = loads.index(min_load)
        avg_load = np.mean(loads)

        report_data.append({
            'Region': region,
            'Peak Load (MW)': peak_load,
            'Peak Load Time (Hour)': peak_load_time,
            'Min Load (MW)': min_load,
            'Min Load Time (Hour)': min_load_time,
            'Avg Load (MW)': avg_load
        })

    # Create DataFrame for report
    report_df = pd.DataFrame(report_data)

    # Display the report table
    st.subheader("Load Report")
    st.dataframe(report_df)

    # Add download option for the report
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="Download Report as CSV",
        data=csv,
        file_name='load_report.csv',
        mime='text/csv',
    )

    # Show the raw data as an optional checkbox
    if st.checkbox("Show raw data"):
        st.write(demand_data)

if __name__ == "__main__":
    main()
