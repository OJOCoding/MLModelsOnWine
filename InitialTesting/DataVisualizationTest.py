import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

w_wine_data = pd.read_csv('DATA/winequality-white.csv', sep=';')
r_wine_data = pd.read_csv('DATA/winequality-red.csv', sep=';')


# Displaying the first few rows of the dataset
print(w_wine_data.head())
print(w_wine_data.info())
print("Shape of white wine dataset: ", w_wine_data.shape)

print(r_wine_data.head())
print("Shape of white wine dataset: ", r_wine_data.shape)
print(r_wine_data.info())

# Checking for missing values
print(w_wine_data.isnull().sum())
print(r_wine_data.isnull().sum())

for column in r_wine_data.columns[:-1]:  # Exclude 'quality' column
    plt.figure(figsize=(8, 4))  # Set figure size
    sns.histplot(r_wine_data[column], kde=True, bins=30)
    plt.title(f"Distribution of {column} (Red Wine)")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

corr_matrix = r_wine_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Red Wine)")
plt.show()

# Data Visualization

for column in r_wine_data.columns[:-1]:  # Exclude 'quality' column
    sns.histplot(r_wine_data[column], kde=True)
    plt.title(f"Distribution of {column} (Red Wine)")
    plt.show()

corr_matrix = r_wine_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Red Wine)")
plt.show()

for column in r_wine_data.columns[:-1]:  # Exclude 'quality' column
    sns.boxplot(x=r_wine_data[column])
    plt.title(f"Boxplot of {column} (Red Wine)")
    plt.show()

for column in w_wine_data.columns[:-1]:  # Exclude 'quality' column
    sns.histplot(w_wine_data[column], kde=True)
    plt.title(f"Distribution of {column} (White Wine)")
    plt.show()

corr_matrix = w_wine_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (White Wine)")
plt.show()

for column in w_wine_data.columns[:-1]:  # Exclude 'quality' column
    sns.boxplot(x=w_wine_data[column])
    plt.title(f"Boxplot of {column} (White Wine)")
    plt.show()


#Standardize Data

scaler = StandardScaler()
red_wine_scaled = scaler.fit_transform(r_wine_data.iloc[:, :-1])
white_wine_scaled = scaler.fit_transform(w_wine_data.iloc[:, :-1])

#Data Comparison

for column in r_wine_data.columns[:-1]:
    sns.kdeplot(r_wine_data[column], label="Red Wine")
    sns.kdeplot(w_wine_data[column], label="White Wine")
    plt.title(f"Comparison of {column}")
    plt.legend()
    plt.show()

sns.countplot(x=r_wine_data['quality'])
plt.title("Quality Distribution (Red Wine)")
plt.show()

sns.countplot(x=w_wine_data['quality'])
plt.title("Quality Distribution (White Wine)")
plt.show()