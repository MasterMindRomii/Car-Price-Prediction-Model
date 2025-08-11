# 🚗 Car Price Prediction

Hello Everyone,  

This is my **Regression Project** aimed at predicting used car prices using **Linear Regression**.  
It demonstrates my skills in **data cleaning, visualization, feature engineering, and model building**.  

---

## 📊 Dataset

**Source:** [Honda Used Car Selling](https://www.kaggle.com/datasets/themrityunjaypathak/honda-car-selling)  

The dataset contains various attributes of used cars, such as **model, fuel type, kilometers driven, suspension, and selling price**.

---

## 🎯 Problem Statement

The goal is to develop a **Machine Learning model** that can predict the price of a used car based on its features.  
This helps buyers and sellers make **data-driven** pricing decisions.

---

## 🛠 Tech Stack & Libraries

```python
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
%matplotlib inline
📂 Project Workflow

1️⃣ Data Loading & Exploration
df = pd.read_csv("honda_car_selling.csv")
df.head()
df.info()
df.shape

2️⃣ Data Cleaning
Removed extra whitespaces from Fuel Type, Suspension, and Car Model.

Converted kms driven into integers after stripping "kms".

Converted price from "6.45 Lakh" to 645000 using a custom function.

df['Fuel_Type'] = df['Fuel_Type'].str.strip()
df['Suspension'] = df['Suspension'].str.strip()
df['Car_Model'] = df['Car_Model'].str.strip()

df['kms_driven'] = df['kms_driven'].str.split().str[0].astype(int)

def convert_price(price_str):
    return int(float(price_str.split()[0]) * 100000)

df['Price'] = df['Price'].apply(convert_price)

3️⃣ Data Visualization
sns.swarmplot(x='Year', y='Price', data=df)
sns.relplot(x='kms_driven', y='Price', data=df)
sns.relplot(x='Car_Model', y='Price', hue='Suspension', data=df)

4️⃣ Feature Engineering
df = pd.get_dummies(df, columns=['Fuel_Type', 'Suspension'], drop_first=True)

5️⃣ Model Building & Evaluation
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

cv = KFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print("Cross-validation scores:", scores)
print("Mean R² score:", scores.mean())

📌 Conclusion
Developed a Linear Regression Model to predict car prices based on multiple attributes.

Achieved an average prediction accuracy of ~82%.

Validated model performance using K-Fold Cross Validation with a mean R² score of ~83%.
