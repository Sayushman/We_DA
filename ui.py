import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("Weather Data Analysis")

# Function to read and preprocess the dataset
def load_data(file):
    weather_dataset = pd.read_csv(file)
    weather_dataset['Formatted Date'] = pd.to_datetime(weather_dataset['Formatted Date'], utc=True)
    weather_dataset['Formatted Date'] = weather_dataset['Formatted Date'].dt.tz_convert(None)

    if len(weather_dataset['Loud Cover'].unique()) == 1:
        weather_dataset.drop(columns='Loud Cover', inplace=True)

    return weather_dataset

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing weather data", type=["csv"])

if uploaded_file:
    # Load and preprocess the data
    weather_dataset = load_data(uploaded_file)

    st.write("### Data Preview")
    st.write(weather_dataset.head())

    # Function to calculate Z-score
    def calculate_z_score(df, column):
        mean = df[column].mean()
        std = df[column].std()
        return (df[column] - mean) / std

    # Function to remove outliers
    def remove_outliers_z_score(df, column):
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        return df[(z_scores >= -3) & (z_scores <= 3)]

    # Calculating the number of outliers in each column
    outliers = {}
    numeric_columns = weather_dataset.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        z_scores = calculate_z_score(weather_dataset, column)
        outliers[column] = weather_dataset[(z_scores < -3) | (z_scores > 3)]
    outlier_counts = {column: len(outliers_df) for column, outliers_df in outliers.items()}

    st.write("### Number of Outliers in Each Column Before Removal")
    st.write(outlier_counts)

    # Removing outliers
    for column in numeric_columns:
        weather_dataset = remove_outliers_z_score(weather_dataset, column)

    # Plotting box plots before and after removing outliers
    st.write("### Box Plots Before and After Removing Outliers")
    fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(15, 5 * len(numeric_columns)))
    for i, column in enumerate(numeric_columns):
        sns.boxplot(y=weather_dataset[column], ax=axes[i, 0])
        axes[i, 0].set_title(f'After: {column}')
    st.pyplot(fig)

    # Dealing with duplicated rows
    st.write(f'The number of duplicated rows is: {weather_dataset.duplicated().sum()}')
    weather_dataset = weather_dataset.drop_duplicates()

    # One-hot encoding for 'Precip Type'
    def one_hot(df):
        df['Precip Type'] = df['Precip Type'].fillna('None')
        df['Rain'] = ((df['Precip Type'] == 'rain').astype(int) & (df['Precip Type'] != 'None')).astype(int)
        df['Snow'] =((df['Precip Type'] == 'snow').astype(int) & (df['Precip Type'] != 'None')).astype(int)
        return df

    weather_dataset = one_hot(weather_dataset)

    # Temperature distribution by precip type
    st.write("### Temperature Distribution by Precipitation Type")
    g = sns.FacetGrid(weather_dataset, col='Precip Type', col_wrap=3, height=4)
    g.map(sns.histplot, 'Temperature (C)', bins=30, kde=False)
    g.fig.suptitle('Temperature Distribution by Precipitation Type', fontsize=16)
    g.fig.subplots_adjust(top=0.9)
    g.set_axis_labels('Temperature (C)', 'Count')
    st.pyplot(g.fig)

    # Histogram of 'Summary' column
    def histogram_for_summary(df):
        threshold = 5000
        summary_data = df['Summary']
        summary_counts = summary_data.value_counts()
        adjusted_summary_counts = {}
        for summary, count in summary_counts.items():
            if count < threshold:
                adjusted_summary_counts['OTHER'] = adjusted_summary_counts.get('OTHER', 0) + count
            else:
                adjusted_summary_counts[summary] = count

        plt.figure(figsize=(10, 6))
        pd.Series(adjusted_summary_counts).plot(kind='bar')
        plt.title('Histogram of Weather Summary (Adjusted)')
        plt.xlabel('Weather Summary')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    st.write("### Histogram of Weather Summary")
    histogram_for_summary(weather_dataset)

    # Dependency of temperature and visibility
    st.write("### Visibility vs Temperature")
    plt.figure(figsize=(10, 6))
    plt.scatter(weather_dataset['Temperature (C)'], weather_dataset['Visibility (km)'], alpha=0.5)
    plt.title('Visibility vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Visibility')
    st.pyplot(plt)

    # Correlation between 'Precip Type' and other factors
    st.write("### Correlation Analysis")

    def correlation_plots(df):
        no_rain_snow = df[(df['Rain'] == 0) & (df['Snow'] == 0)]

        # Visibility
        plt.figure(figsize=(10, 6))
        sns.boxplot(y='Visibility (km)', data=no_rain_snow)
        plt.title('Visibility Distribution When There is No Rain and No Snow')
        st.pyplot(plt)

        # Temperature and Apparent Temperature
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y='Apparent Temperature (C)', data=no_rain_snow)
        plt.subplot(1, 2, 2)
        sns.boxplot(y='Temperature (C)', data=no_rain_snow)
        st.pyplot(plt)

        # Humidity
        plt.figure(figsize=(10, 6))
        sns.boxplot(y='Humidity', data=no_rain_snow)
        st.pyplot(plt)

        # Wind Speed and Wind Bearing
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y='Wind Bearing (degrees)', data=no_rain_snow)
        plt.subplot(1, 2, 2)
        sns.boxplot(y='Wind Speed (km/h)', data=no_rain_snow)
        st.pyplot(plt)

        # Pressure
        plt.figure(figsize=(10, 6))
        sns.boxplot(y='Pressure (millibars)', data=no_rain_snow)
        st.pyplot(plt)

    correlation_plots(weather_dataset)

    # Monthly temperature for a specific year
    st.write("### Average Monthly Temperatures for a Specific Year")
    year = st.slider('Select Year', int(weather_dataset['Formatted Date'].dt.year.min()), int(weather_dataset['Formatted Date'].dt.year.max()), 2008)

    def plot_monthly_temperature_for_year(df, year):
        year_data = df[df['Formatted Date'].dt.year == year]
        monthly_avg = year_data.groupby(year_data['Formatted Date'].dt.month)['Temperature (C)'].mean()

        plt.figure(figsize=(12, 6))
        monthly_avg.plot(kind='bar', color='skyblue')

        plt.title(f'Average Monthly Temperatures for {year}')
        plt.xlabel('Month')
        plt.ylabel('Average Temperature (C)')
        plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        st.pyplot(plt)

    plot_monthly_temperature_for_year(weather_dataset, year)

    # Yearly average temperature
    st.write("### Yearly Average Temperatures")
    def plot_for_comparing_temperature(df):
        yearly_avg = df.groupby(df['Formatted Date'].dt.year)['Temperature (C)'].mean()

        yearly_avg = yearly_avg[yearly_avg.index != 2005]
        plt.figure(figsize=(10, 5))
        yearly_avg.plot(kind='bar', color='lightblue')
        plt.title('Yearly Average Temperatures')
        plt.xlabel('Year')
        plt.ylabel('Average Temperature')
        plt.xticks(rotation=0)
        st.pyplot(plt)

    plot_for_comparing_temperature(weather_dataset)

    # Monthly humidity for a specific year
    st.write("### Average Monthly Humidity for a Specific Year")
    year = st.slider('Select Year for Humidity', int(weather_dataset['Formatted Date'].dt.year.min()), int(weather_dataset['Formatted Date'].dt.year.max()), 2007)

    def plot_monthly_humidity_for_year(df, year):
        year_data = df[df['Formatted Date'].dt.year == year]
        monthly_avg = year_data.groupby(year_data['Formatted Date'].dt.month)['Humidity'].mean()

        plt.figure(figsize=(12, 6))
        monthly_avg.plot(kind='bar', color='skyblue')

        plt.title(f'Average Monthly Humidity for {year}')
        plt.xlabel('Month')
        plt.ylabel('Humidity')
        plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        st.pyplot(plt)

    plot_monthly_humidity_for_year(weather_dataset, year)
else:
    st.write("Please upload a CSV file to proceed.")
