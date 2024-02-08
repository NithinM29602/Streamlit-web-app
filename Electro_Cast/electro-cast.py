import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX 

# Function to predict electricity production
def predict_electricity_production(year, month):
    # Define the start date
    start_date = datetime(2018, 1, 1)

    # Ensure the user has entered a date after January 2018
    end_date = datetime(year, month, 1)

    # Calculate the number of months between the start and end dates
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # Read the data from CSV file
    data = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'])
    data.set_index('DATE', inplace=True)
    data.dropna(inplace=True)

    # Fit the SARIMA model
    model = SARIMAX(data['Value'], order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
    fitted_model = model.fit()

    # Make predictions
    predictions = fitted_model.predict(start=data.index[-1], end=data.index[-1] + pd.DateOffset(months=num_months))

    # Visualize the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data['Value'], label='Original')
    plt.plot(predictions, color='red', label='Predictions')
    plt.xlabel('Year')
    plt.ylabel('Electric Production')
    plt.title('Electric Production Forecast')
    plt.legend()
    
    st.pyplot(plt)

    # Return the last predicted value
    return predictions[-1]

def main():
    st.title("Electricity Production Predictor")
    st.sidebar.title("Done by Shit")  # Sidebar title
    year = st.selectbox("Select Year", list(range(2018, datetime.now().year + 30)))
    month = st.selectbox("Select Month", list(range(1, 13)))
    
    if st.button("Predict"):
        bill_amount = predict_electricity_production(year, month)
        st.success(f"The predicted electricity production is {bill_amount:.3f} billion kwh")
    
if __name__ == "__main__":
    main()
