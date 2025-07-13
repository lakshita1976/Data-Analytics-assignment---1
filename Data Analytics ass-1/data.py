# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📂 Load Dataset
df = pd.read_csv("user_behavior.csv")  # Change filename if needed

# 👁 Initial Exploration
print("🔍 Dataset shape:", df.shape)
print("🧱 Columns:", df.columns.tolist())
print("📊 Data types:\n", df.dtypes)
print("❓ Missing values:\n", df.isnull().sum())
print("🔁 Duplicate rows:", df.duplicated().sum())
print("👀 First few rows:\n", df.head())

# 🧹 Data Cleaning

# Drop columns with more than 50% missing values
df = df[df.columns[df.isnull().mean() < 0.5]]

# Remove duplicates
df = df.drop_duplicates()

# 🧽 Fill missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 🔡 Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 🗑 Drop non-useful columns (like IDs or timestamps)
df = df.drop(columns=['user_id', 'timestamp'], errors='ignore')

# 📏 Feature Scaling
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# 📊 EDA Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Optional: Plot one column (if available)
if "age" in df.columns:
    sns.histplot(df["age"], kde=True)
    plt.title("Age Distribution")
    plt.show()

# 🎯 Train-Test Split (if target available)
target = "purchase" if "purchase" in df.columns else None
if target:
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Train-test split done")
else:
    print("⚠️ No target variable found. Dataset ready for clustering or analysis.")

# 🧾 Final Data Snapshot
print("📌 Final dataset shape:", df.shape)
print("📈 Summary statistics:\n", df.describe())
