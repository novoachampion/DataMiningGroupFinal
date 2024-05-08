import pandas as pd
import os
import requests

def download_and_save_excel(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded and saved {filename}")

def excel_to_csvs(excel_path, csv_dir='csv'):
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    xls = pd.ExcelFile(excel_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=1)  # Headers are on the second row
        df.columns = df.columns.str.strip()  # Normalize the column names by stripping whitespace
        sheet_name_hyphenated = sheet_name.replace(' ', '_')
        csv_file_path = os.path.join(csv_dir, f"{sheet_name_hyphenated}.csv")
        df.to_csv(csv_file_path, index=False)
        print(f"Processed and saved '{sheet_name}' as '{csv_file_path}'")

def load_data(csv_file_path, header_row=1):
    df = pd.read_csv(csv_file_path, header=header_row)
    df.columns = df.columns.str.strip()  # Strip spaces to ensure consistent column names
    return df

def load_multiple_data(files, header_row=1):
    data_frames = [load_data(file, header_row) for file in files]  # Use load_data to ensure consistency
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def prepare_features_targets(df, features_cols, targets_cols):
    X = df.drop(targets_cols, axis=1)
    y = df[targets_cols]
    return X, y
