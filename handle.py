import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column
iris_df['target'] = iris.target

# Define the preprocessing function
def preprocess_dataset(df):
    # Introduce missing values (NaN) in every 10th entry of the first column
    df.iloc[::10, 0] = float('NaN')

    # Initialize the imputer to replace NaN values with the mean
    imputer = SimpleImputer(strategy='mean')

    # Apply the imputer to fill missing values
    df[df.columns] = imputer.fit_transform(df[df.columns])

    # Initialize the scaler to standardize the feature columns (excluding 'target')
    scaler = StandardScaler()

    # Apply the scaler to all feature columns
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    return df

# Preprocess the dataset
preprocessed_df = preprocess_dataset(iris_df)

# Print the preprocessed dataset
print("Preprocessed dataset:")
print(preprocessed_df.head())
