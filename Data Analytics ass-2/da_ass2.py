# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Step 2: Download CSV Manually from:
# https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
# Save it as: iris.csv in your project folder

df = pd.read_csv("iris.csv")

# 👣 Step 3: Preprocessing
X = df.drop('species', axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔁 Step 4: Train/Test with Various Ratios
split_ratios = [0.7, 0.6, 0.8]
for ratio in split_ratios:
    print(f"\n=== Train/Test Split: {int(ratio*100)}/{int((1-ratio)*100)} ===")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=1-ratio, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Cross Entropy Loss:", log_loss(y_test, y_pred_prob))

# 🤖 Step 5: Try Multiple ML Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

for name, model in models.items():
    print(f"\n=== Model: {name} ===")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Cross Entropy Loss:", log_loss(y_test, y_prob))

    # Optional: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 🔟 Step 6: 10-Fold Cross Validation
print("\n=== 10-Fold Cross Validation ===")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=kfold, scoring='accuracy')
    print(f"{name} - Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
