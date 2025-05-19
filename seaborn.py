import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\Student\Desktop\harishree\iris.csv")

# Pairplot of the dataset
sns.pairplot(df)
plt.title("Pairplot of the Dataset")
plt.show()

# Check if the first column is categorical
if df.iloc[:, 0].dtype == 'object':
    sns.countplot(x=df.columns[0], data=df)
    plt.title("Bar Chart of Categorical Column")
    plt.xlabel(df.columns[0])
    plt.ylabel("Count")
    plt.show()
else:
    print("No categorical column found to plot bar chart.")
