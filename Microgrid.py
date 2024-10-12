import streamlit as st
import pandas as pd

# Load datasets
demand_data = pd.read_csv('delhi_electricity_demand_hourly.csv')
weather_data = pd.read_csv('delhipastweather.csv')

st.title("Microgrid Simulation Dashboard")

# User Inputs
solar_panels = st.text_input("Enter number of Solar Panels", "10")  # Default value of 10
wind_turbines = st.text_input("Enter number of Wind Turbines", "5")  # Default value of 5
batteries = st.number_input("Enter number of Batteries", min_value=0, value=3)  # Number input for batteries

# Run simulation only if inputs are valid
if st.button("Run Simulation"):
    try:
        # Convert text inputs to integers
        solar_panels = int(solar_panels)
        wind_turbines = int(wind_turbines)

        # Run simulation logic
        generated_energy_solar = solar_panels * 5  # kWh per day
        generated_energy_wind = wind_turbines * 7   # kWh per day
        total_generated_energy = generated_energy_solar + generated_energy_wind

        stored_energy = batteries * 10  # kWh
        average_demand = demand_data['DELHI'].mean()  # Example average demand

        optimal_energy_usage = min(stored_energy + total_generated_energy, 100)

        st.write(f"Total Generated Energy: {total_generated_energy} kWh")
        st.write(f"Stored Energy: {stored_energy} kWh")
        st.write(f"Optimal Energy Usage: {optimal_energy_usage} kWh")
        st.write(f"Average Demand: {average_demand} kWh")

        # Visualization or other features can be added here.

    except ValueError:
        st.error("Please enter valid numerical inputs.")
