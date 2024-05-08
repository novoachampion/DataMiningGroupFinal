import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# URL of the Excel file on GitHub
excel_url = 'https://github.com/novoachampion/DataMiningGroupFinal/raw/fcd2288b9e94e4aac1b8fac5b3c6179cc02eda85/stockPortfolio.xlsx'

# Download the Excel file
response = requests.get(excel_url)
with open('stockPortfolio.xlsx', 'wb') as f:
    f.write(response.content)

# Create a directory named "csv" if it doesn't exist
csv_dir = 'csv'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Load the Excel file
xls = pd.ExcelFile('stockPortfolio.xlsx')

# Process each sheet and save as CSV
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Replace spaces with hyphens in sheet_name
    sheet_name_hyphenated = sheet_name.replace(' ', '_')
    
    # Define the CSV file path with hyphenated name
    csv_file_path = os.path.join(csv_dir, f"{sheet_name_hyphenated}.csv")
    
    # Save the DataFrame to CSV
    df.to_csv(csv_file_path, index=False)
    
    print(f"Processed and saved '{sheet_name}' as '{csv_file_path}'")

csv_file_path = 'csv/1st_period.csv'  # Replace 'example_sheet.csv' with the actual file name

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, header=1)

# Get descriptive statistics of the DataFrame
description = df.describe()

# Print the description
print(description)


# Print column names
print(df.columns)

# Select features and target
X = df.drop(['Systematic Risk', 'Total Risk'], axis=1)
y = df[['Systematic Risk', 'Total Risk']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared score: {r2}")

    return model, predictions

# Call the function
model, predictions = train_and_evaluate(X_train, y_train, X_test, y_test)
