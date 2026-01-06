import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('/Users/hiten/Documents/AIML task/dataset.csv')

# Convert TotalCharges to numeric, coercing errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values
print(df.isnull().sum())

# Handle missing values (fill with median or drop)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(8, 5))
df['Churn'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

# 2. Churn vs Tenure
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure Distribution by Churn')
plt.tight_layout()
plt.savefig('churn_vs_tenure.png')
plt.show()

# 3. Churn vs Contract Type
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', figsize=(10, 6), color=['#2ecc71', '#e74c3c'])
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Percentage')
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('churn_vs_contract.png')
plt.show()

# 4. Correlation heatmap for numeric features
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Scatter plot - shows relationship between two numeric variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, alpha=0.6)
plt.title('Monthly Charges vs Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Monthly Charges ($)')
plt.legend(title='Churn')
plt.tight_layout()
plt.savefig('charges_vs_tenure.png')
plt.show()

# Generate statistical summary
summary = df.describe()
summary.to_csv('statistical_summary.csv')

# Churn rate by different features
print("\nChurn Rate by Contract Type:")
print(df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100)

print("\nChurn Rate by Internet Service:")
print(df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100)

# Export cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
