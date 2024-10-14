import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load datasets
demand_data = pd.read_csv('delhi_electricity_demand (1).csv')
weather_data = pd.read_csv('delhipastweather.csv')

# Merge datasets on date and time
demand_data['DateTime'] = pd.to_datetime(demand_data['Date'] + ' ' + demand_data['TimeSlot'])
weather_data['DateTime'] = pd.to_datetime(weather_data['date'] + ' ' + weather_data['time'])
merged_data = pd.merge(demand_data, weather_data, on='DateTime')

# Function to generate recommendations
def generate_recommendations(peak_hours, off_peak_hours):
    recommendations = []
    for hour in off_peak_hours:
        recommendations.append(f"Consider running energy-intensive appliances between {hour}:00 and {hour + 1}:00 to save on energy costs.")
    for hour in peak_hours:
        recommendations.append(f"Avoid using heavy appliances during {hour}:00 to {hour + 1}:00, as demand is high.")
    return recommendations

# Streamlit app layout
st.title("Delhi Electricity Demand and Weather Analysis")
st.write("This application analyzes electricity demand patterns based on historical data and weather conditions.")

# Date selection
start_date = st.date_input("Select start date:", value=pd.to_datetime(merged_data['DateTime'].min()))
end_date = st.date_input("Select end date:", value=pd.to_datetime(merged_data['DateTime'].max()))

# Filter data based on selected date range
filtered_data = merged_data[(merged_data['DateTime'] >= pd.to_datetime(start_date)) & 
                             (merged_data['DateTime'] <= pd.to_datetime(end_date))]

# Ensure there is data to analyze
if filtered_data.empty:
    st.write("No data available for the selected date range.")
else:
    # Feature Engineering
    filtered_data['hour'] = filtered_data['DateTime'].dt.hour

    # Demand Pattern Analysis
    daily_load_profile = filtered_data.groupby('hour').agg({'DELHI': 'mean'}).reset_index()

    # Create a complete DataFrame for 24 hours
    complete_hours = pd.DataFrame({'hour': range(24)})
    daily_load_profile = pd.merge(complete_hours, daily_load_profile, on='hour', how='left')

    # Remove missing values (drop NaN)
    daily_load_profile = daily_load_profile.dropna(subset=['DELHI'])

    # Plot the daily load profile
    st.subheader('Daily Load Profile for Selected Date Range')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=daily_load_profile['hour'], y=daily_load_profile['DELHI'], marker='o', color='blue', ax=ax)
    ax.set_title('Daily Load Profile for DELHI')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Load (MW)')
    ax.set_xticks(range(24))
    ax.grid()
    st.pyplot(fig)

    # Calculate Peak and Off-Peak Hours
    top_peak_hours = daily_load_profile.nlargest(4, 'DELHI')['hour'].values
    bottom_peak_hours = daily_load_profile.nsmallest(4, 'DELHI')['hour'].values

    average_peak_demand = daily_load_profile[daily_load_profile['hour'].isin(top_peak_hours)]['DELHI'].mean()
    average_off_peak_demand = daily_load_profile[daily_load_profile['hour'].isin(bottom_peak_hours)]['DELHI'].mean()

    # Display Peak and Off-Peak Analysis
    st.write(f"Top 4 Peak Hours: {top_peak_hours}, Average Peak Demand: {average_peak_demand:.2f} MW")
    st.write(f"Bottom 4 Off-Peak Hours: {bottom_peak_hours}, Average Off-Peak Demand: {average_off_peak_demand:.2f} MW")

    # Generate recommendations
    recommendations = generate_recommendations(top_peak_hours, bottom_peak_hours)

    st.subheader("Recommendations for Consumers")
    for rec in recommendations:
        st.write(rec)

    # Optional: Visualize Peak and Off-Peak Hours
    st.subheader('Peak and Off-Peak Demand Analysis')
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Plotting peak hours
    ax2.bar(top_peak_hours, daily_load_profile[daily_load_profile['hour'].isin(top_peak_hours)]['DELHI'], 
            color='red', alpha=0.6, label='Peak Demand Hours')
    
    # Plotting off-peak hours
    ax2.bar(bottom_peak_hours, daily_load_profile[daily_load_profile['hour'].isin(bottom_peak_hours)]['DELHI'], 
            color='green', alpha=0.6, label='Off-Peak Demand Hours')
    
    # Average demand lines
    ax2.axhline(y=average_peak_demand, color='orange', linestyle='--', label='Average Peak Demand')
    ax2.axhline(y=average_off_peak_demand, color='blue', linestyle='--', label='Average Off-Peak Demand')

    # Additional plot formatting
    ax2.set_title('Peak and Off-Peak Demand Analysis for DELHI Load')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Load (MW)')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24))
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)
