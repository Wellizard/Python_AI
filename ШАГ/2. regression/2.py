import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv(r'ШАГ\2. regression\energy_usage_plus.csv')

# ==========================================
# Завдання 1
# ==========================================
X1 = df[['temperature', 'humidity', 'hour', 'is_weekend']]
y = df['consumption']

X_train1, X_test1, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=42)

model1 = LinearRegression()
model1.fit(X_train1, y_train)

y_pred1 = model1.predict(X_test1)
mape1 = mean_absolute_percentage_error(y_test, y_pred1) * 100
print(f"Похибка базової моделі (Завдання 1): {mape1:.2f}%")

# ==========================================
# Завдання 2
# ==========================================
X2 = df[['temperature', 'humidity', 'hour', 'is_weekend', 'season', 'district_type']]

X2_encoded = pd.get_dummies(X2, columns=['season', 'district_type'], drop_first=True)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_encoded, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)
mape2 = mean_absolute_percentage_error(y_test2, y_pred2) * 100
print(f"Похибка покращеної моделі (Завдання 2): {mape2:.2f}%")

# ==========================================
# Побудова графіка
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test2, y=y_pred2, alpha=0.6, color='blue', label='Прогнозовані значення')

plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 
         color='red', linestyle='--', label='Ідеальний прогноз')

plt.title(f'Справжнє vs Прогнозоване споживання (Похибка: {mape2:.2f}%)')
plt.xlabel('Справжнє споживання (кВт·год)')
plt.ylabel('Прогнозоване споживання (кВт·год)')
plt.legend()
plt.grid(True)
plt.show()