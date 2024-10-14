import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
demand_data = pd.read_csv('delhi_electricity_demand_hourly.csv')
weather_data = pd.read_csv('delhipastweather.csv')

# Function to dynamically calculate sunlight hours based on weather data
def calculate_sunlight_hours(temperature):
    return max(0, 12 - (temperature - 25) / 5)

# Extract average sunlight hours and wind speed from weather data
sunlight_hours = calculate_sunlight_hours(weather_data['temperature_2m'].mean())
average_wind_speed = weather_data['wind_speed_10m'].mean()

# Add a sidebar for inputs
st.sidebar.title("Simulation Parameters")
st.sidebar.write("Adjust the parameters to simulate energy generation and demand.")

# User Inputs with detailed explanation
solar_panels = st.sidebar.number_input("Enter number of Solar Panels", min_value=1, value=10, help="Number of solar panels installed in the system.")
solar_panel_capacity = st.sidebar.number_input("Enter Solar Panel Capacity (W)", min_value=0, value=300, help="Capacity of each solar panel in watts.")
wind_turbines = st.sidebar.number_input("Enter number of Wind Turbines", min_value=1, value=5, help="Number of wind turbines installed.")
wind_turbine_diameter = st.sidebar.number_input("Enter Wind Turbine Diameter (m)", min_value=0.0, value=2.0, help="Diameter of each wind turbine's rotor.")
batteries = st.sidebar.number_input("Enter number of Batteries", min_value=0, value=3, help="Number of batteries available for energy storage.")
battery_capacity = 10  # Assuming each battery can store 10 kWh

# Additional dynamic inputs for wind and solar data
# Additional dynamic inputs for wind and solar data
use_dynamic_weather = st.sidebar.checkbox("Use Dynamic Weather Data", help="Check this if you want to input custom sunlight and wind data for the day.")

# Function to generate dynamic weather data
def generate_dynamic_weather(total_sunlight_hours, avg_wind_speed):
    sunlight_hours = np.zeros(24)
    wind_speeds = np.zeros(24)
    
    # Sunlight hours distribution (assume sunlight between 6 AM to 6 PM)
    peak_sun_hours = total_sunlight_hours / 12 if total_sunlight_hours <= 12 else 1
    for i in range(6, 18):
        sunlight_hours[i] = peak_sun_hours
    
    # Dynamic wind speed distribution
    for i in range(24):
        variation = np.random.uniform(-1, 1)  # Random variation between -1 and 1
        wind_speeds[i] = max(0, avg_wind_speed + variation)
    
    return sunlight_hours, wind_speeds

# If dynamic weather is enabled, show additional sliders for sunlight and wind speed
if use_dynamic_weather:
    total_sunlight_hours = st.sidebar.slider("Total Sunlight Hours", min_value=0, max_value=12, value=8)
    avg_wind_speed = st.sidebar.slider("Average Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=5.0)
else:
    # Use default values from the weather data
    total_sunlight_hours = sunlight_hours
    avg_wind_speed = average_wind_speed

# Generate sunlight hours and wind speeds
sunlight_hours, wind_speeds = generate_dynamic_weather(total_sunlight_hours, avg_wind_speed)

st.title("Enhanced Microgrid Simulation Dashboard")

# Run simulation only if inputs are valid
if st.button("Run Simulation"):
    try:
        # Solar energy generation
        power_output_solar = solar_panels * (solar_panel_capacity / 1000)  # Convert W to kW
        energy_generated_solar = np.array([power_output_solar * sun for sun in sunlight_hours])  # Hourly solar energy

        # Wind energy generation
        air_density = 1.225  # kg/m³
        swept_area = np.pi * (wind_turbine_diameter / 2) ** 2  # m²
        power_output_wind = np.array([0.5 * air_density * swept_area * (wind ** 3) / 1000 for wind in wind_speeds])  # Hourly wind energy

        # Total energy generated over 24 hours
        total_generated_energy = energy_generated_solar + power_output_wind

        # Battery storage capacity
        stored_energy = min(batteries * battery_capacity, total_generated_energy.sum())  # Can't store more than battery capacity

        # Energy demand for DELHI region
        hourly_demand = demand_data['DELHI'][:24].values
        average_demand = hourly_demand.mean()

        # Calculate surplus and deficits
        surplus_energy = total_generated_energy - hourly_demand
        deficit_energy = hourly_demand - total_generated_energy
        surplus_energy[surplus_energy < 0] = 0
        deficit_energy[deficit_energy < 0] = 0

        # Display Results
        st.markdown("### Simulation Results")
        st.write(f"**Total Generated Energy from Solar Panels:** {energy_generated_solar.sum():.2f} kWh")
        st.write(f"**Total Generated Energy from Wind Turbines:** {power_output_wind.sum():.2f} kWh")
        st.write(f"**Total Generated Energy:** {total_generated_energy.sum():.2f} kWh")
        st.write(f"**Stored Energy:** {stored_energy:.2f} kWh")
        st.write(f"**Average Demand:** {average_demand:.2f} kWh")
        
        # Efficiency Metrics
        energy_met = np.sum(total_generated_energy >= hourly_demand) / 24 * 100  # Percentage of hours when generation meets demand
        st.write(f"**Percentage of time demand met:** {energy_met:.2f}%")

        # Plotting generated energy vs demand
        plt.figure(figsize=(10, 5))
        plt.plot(hourly_demand, label='Demand (kWh)', color='red', linestyle='--', marker='o')
        plt.plot(total_generated_energy, label='Generated Energy (kWh)', color='green', linestyle='-', marker='o')
        plt.fill_between(np.arange(24), hourly_demand, total_generated_energy, where=total_generated_energy > hourly_demand, 
                         interpolate=True, color='green', alpha=0.3, label='Surplus Energy')
        plt.fill_between(np.arange(24), hourly_demand, total_generated_energy, where=hourly_demand > total_generated_energy, 
                         interpolate=True, color='red', alpha=0.3, label='Deficit Energy')
        plt.title('Energy Demand vs Generated Energy Over 24 Hours')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Energy (kWh)')
        plt.xticks(ticks=np.arange(0, 24), labels=[f"{i}:00" for i in range(24)], rotation=45)
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Allow user to download simulation report
        report = pd.DataFrame({
            "Hour": np.arange(1, 25),
            "Demand (kWh)": hourly_demand,
            "Generated Energy (kWh)": total_generated_energy,
            "Surplus Energy (kWh)": surplus_energy,
            "Deficit Energy (kWh)": deficit_energy
        })
        st.markdown("### Download Simulation Report")
        st.write(report)
        csv = report.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download as CSV", data=csv, file_name='microgrid_simulation_report.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error: {e}")

