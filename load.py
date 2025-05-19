import pandas as pd

def explore_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return

    print("Dataset information:")
    print(df.info())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nUnique values for categorical columns:")
    for column in df.select_dtypes(include='object').columns:
        print(f"{column}: {df[column].unique()}")

# Specify the correct file path to your CSV or Excel file
file_path = r'C:/Users/Student/Desktop/harishree/iris.csv'  # Raw string to handle backslashes in file path

explore_dataset(file_path)
