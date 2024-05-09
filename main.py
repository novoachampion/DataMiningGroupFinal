import pandas as pd
from data_preparation import download_and_save_excel, excel_to_csvs, load_data, prepare_features_targets
from model import train_and_evaluate
from sklearn.metrics import mean_squared_error, r2_score
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors

def create_excel_report(file_name, df, model_metrics, correlation_matrix, descriptive_stats):
    with pd.ExcelWriter(file_name) as writer:
        # Save the first few rows of the DataFrame to a sheet
        df.head().to_excel(writer, sheet_name='DataFrame_Head', index=False)
        
        # Save the model metrics to a sheet
        metrics_df = pd.DataFrame([model_metrics])
        metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        
        # Save the correlation matrix to a sheet
        correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        
        # Save the descriptive statistics to a sheet
        descriptive_stats.to_excel(writer, sheet_name='Descriptive_Statistics')

def main():
    url = 'https://github.com/novoachampion/DataMiningGroupFinal/raw/fcd2288b9e94e4aac1b8fac5b3c6179cc02eda85/stockPortfolio.xlsx'
    filename = 'stockPortfolio.xlsx'
    csv_dir = 'csv'
    all_time_csv = 'csv/all_period.csv'  # Path to the all-time CSV file
    report_file = 'Data_Analysis_Report.xlsx'  # Excel file name
    
    # Download and process the Excel file
    print("Downloading and processing Excel file...")
    download_and_save_excel(url, filename)
    excel_to_csvs(filename, csv_dir)
    
    # Load the all-time data from a single CSV file
    print("Loading data from all_period.csv...")
    df = load_data(all_time_csv)
    
    # Original column names based on your dataset
    original_columns = ['ID', 'Large_BP', 'Large_ROE', 'Large_SP', 'Large_Return_Rate_Last_Quarter', 
                        'Large_Market_Value', 'Small_Systematic_Risk', 'Annual_Return', 'Excess_Return', 
                        'Systematic_Risk', 'Total_Risk', 'Abs_Win_Rate', 'Rel_Win_Rate', 
                        'Annual_Return_Normalized', 'Excess_Return_Normalized', 'Systematic_Risk_Normalized', 
                        'Total_Risk_Normalized', 'Abs_Win_Rate_Normalized', 'Rel_Win_Rate_Normalized']
    
    # Update DataFrame with original column names
    df.columns = original_columns
    
    # Print column names and data types to verify correct processing
    print("Updated column names in the DataFrame:", df.columns.tolist())
    print("Data types of the DataFrame:")
    print(df.dtypes)
    
    # Print first few rows to inspect the actual data
    print("First few rows of the DataFrame:\n", df.head())

    # Descriptive statistics
    descriptive_stats = df.describe()
    print("Descriptive statistics of the DataFrame:\n", descriptive_stats)

    # Define features and target column names
    features_cols = original_columns[:-2]  # All columns except the last two as features
    targets_cols = original_columns[-2:]   # The last two columns as targets
    
    print("Features columns:", features_cols)
    print("Target columns:", targets_cols)

    # Prepare features and targets
    print("Preparing features and targets...")
    X, y = prepare_features_targets(df, features_cols, targets_cols)
    print("Shape of features (X):", X.shape)
    print("Shape of targets (y):", y.shape)
    
    # Train the model and evaluate it
    print("Starting model training and evaluation...")
    model, y_test, y_pred, precision, recall, f1 = train_and_evaluate(X, y)
    
    # Print shapes for debugging
    print("Shape of predictions:", y_pred.shape)

    # Distribution of binary predictions
    y_pred_class = (y_pred > y_pred.mean(axis=0)).astype(int)
    print("Distribution of binary predictions:", pd.Series(y_pred_class.flatten()).value_counts())

    # Check correlations
    correlation_matrix = df.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Create Excel report
    print("Generating Excel report...")
    model_metrics = {
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'R-squared Score': r2_score(y_test, y_pred),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    create_excel_report(report_file, df, model_metrics, correlation_matrix, descriptive_stats)
    print(f"Excel report generated: {report_file}")

if __name__ == "__main__":
    main()
