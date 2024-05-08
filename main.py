import pandas as pd
from data_preparation import download_and_save_excel, excel_to_csvs, load_data, prepare_features_targets
from model import train_and_evaluate
from visualize import plot_predictions

def main():
    url = 'https://github.com/novoachampion/DataMiningGroupFinal/raw/fcd2288b9e94e4aac1b8fac5b3c6179cc02eda85/stockPortfolio.xlsx'
    filename = 'stockPortfolio.xlsx'
    csv_dir = 'csv'
    all_time_csv = 'csv/all_period.csv'  # Path to the all-time CSV file
    
    # Download and process the Excel file
    download_and_save_excel(url, filename)
    excel_to_csvs(filename, csv_dir)
    
    # Load the all-time data from a single CSV file
    df = load_data(all_time_csv)
    
    # Ensure that column names are properly formatted (removing any spaces)
    df.columns = df.columns.str.strip()

    # Print column names to verify correct processing
    print("Column names in the DataFrame:", df.columns.tolist())

    # Prepare features and targets
    # Ensure the target columns are correctly named as they appear in the DataFrame
    features_cols = df.columns.drop(['Systematic Risk', 'Total Risk'])  # These must match exactly
    targets_cols = ['Systematic Risk', 'Total Risk']
    X, y = prepare_features_targets(df, features_cols, targets_cols)
    
    # Train the model and evaluate it
    model, predictions = train_and_evaluate(X, y)
    
    # Plot the predictions against actual values
    plot_predictions(y, predictions, 'Risk Analysis')

if __name__ == "__main__":
    main()
