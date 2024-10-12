import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
from io import BytesIO

# Load the data with caching using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('delhi_electricity_demand_hourly.csv')

# Load the demand data
demand_data = load_data()

# Set the title of the Streamlit app
st.title("Delhi Electricity Load Curve and Report")

# Create date input for user to select the desired date
date_input = st.date_input("Select a Date")

# Extract day, month, and year from the selected date
day = date_input.day
month = date_input.month
year = date_input.year

# Filter the data based on the selected date
filtered_data = demand_data[
    (pd.to_datetime(demand_data['Date']).dt.day == day) &
    (pd.to_datetime(demand_data['Date']).dt.month == month) &
    (pd.to_datetime(demand_data['Date']).dt.year == year)
]

# If no data is available for the selected date, show a warning
if filtered_data.empty:
    st.warning(f"No data available for {date_input.strftime('%Y-%m-%d')}. Please choose another date.")
else:
    # Create a selectbox for the user to select the region
    region = st.selectbox("Select a region", ["DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES"])

    # Plot the load curve for the selected region
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['TimeSlot'], filtered_data[region], marker='o')
    plt.title(f"Load Curve for {region} on {date_input.strftime('%Y-%m-%d')}")
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Load (MW)")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Generate the report for the selected region
    peak_load = filtered_data[region].max()
    peak_load_time = filtered_data.loc[filtered_data[region].idxmax(), 'TimeSlot']
    min_load = filtered_data[region].min()
    min_load_time = filtered_data.loc[filtered_data[region].idxmin(), 'TimeSlot']
    avg_load = filtered_data[region].mean()

    # Create a report table
    report_data = {
        "Region": [region],
        "Peak Load (MW)": [peak_load],
        "Peak Load Time": [peak_load_time],
        "Min Load (MW)": [min_load],
        "Min Load Time": [min_load_time],
        "Avg Load (MW)": [avg_load]
    }

    report_df = pd.DataFrame(report_data)

    # Display the report table in Streamlit
    st.subheader("Load Report")
    st.dataframe(report_df)

    # Allow user to download the report
    def download_report(report_df):
        output = io.BytesIO()
    
    # Use the 'xlsxwriter' engine to create the Excel file
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Write the dataframe to the Excel file
        report_df.to_excel(writer, index=False, sheet_name='Load Report')
    
    # Close the writer to save the Excel file
        writer.close()

    # Get the Excel data
        processed_data = output.getvalue()

        return processed_data

    # Provide download button for report
    excel_data = download_report(report_df)
    st.download_button(label="Download Report as Excel",
                       data=excel_data,
                       file_name=f'load_report_{region}_{date_input.strftime("%Y-%m-%d")}.xlsx',
                       mime='application/vnd.ms-excel')

# Show the raw data as an optional checkbox
if st.checkbox("Show raw data"):
    st.write(filtered_data)
