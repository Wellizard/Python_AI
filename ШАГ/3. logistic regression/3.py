import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'ШАГ\3. logistic regression\internship_candidates_cefr_final.csv')

eng_mapping = {
    'Elementary': 1,
    'Pre-Intermediate': 2,
    'Intermediate': 3,
    'Upper-Intermediate': 4,
    'Advanced': 5
}
df['EnglishLevel_Num'] = df['EnglishLevel'].map(eng_mapping)

X = df[['Experience', 'Grade', 'EnglishLevel_Num', 'Age', 'EntryTestScore']]
y = df['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оцінка точності моделі (Accuracy)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Точність моделі (Accuracy): {acc:.2f}")

# ==============================================================
# Побудова графіка ймовірності прийняття
# ==============================================================

test_scores = np.linspace(df['EntryTestScore'].min(), df['EntryTestScore'].max(), 100)
eng_levels = np.array([1, 2, 3, 4, 5])
xx, yy = np.meshgrid(test_scores, eng_levels)

exp_mean = df['Experience'].mean()
grade_mean = df['Grade'].mean()
age_mean = df['Age'].mean()

grid_data = pd.DataFrame({
    'Experience': exp_mean,
    'Grade': grade_mean,
    'EnglishLevel_Num': yy.ravel(),
    'Age': age_mean,
    'EntryTestScore': xx.ravel()
})

probs = model.predict_proba(grid_data)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 6))

contour = plt.contourf(xx, yy, probs, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Ймовірність прийняття (0 - 1)')

scatter = plt.scatter(df['EntryTestScore'], df['EnglishLevel_Num'], 
                      c=df['Accepted'], cmap='bwr', edgecolor='k', alpha=0.6, label='Дані кандидатів')

plt.xlabel('Бали за вступний тест (EntryTestScore)')
plt.ylabel('Рівень англійської (1=Elem, 5=Adv)')
plt.title('Ймовірність прийняття від балу за тест та рівня англійської')
plt.yticks([1, 2, 3, 4, 5], 
           ['Elementary (1)', 'Pre-Int (2)', 'Intermediate (3)', 'Upper-Int (4)', 'Advanced (5)'])
plt.grid(alpha=0.3)
plt.show()