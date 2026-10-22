import numpy as np
import matplotlib.pyplot as plt

# 1
x1 = np.linspace(-10, 10, 500)
y1 = (x1**2) * np.sin(x1)

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, color='blue')
plt.title("Графік функції f(x) = x^2 * sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# 2
# data_norm = np.random.normal(loc=5, scale=2, size=1000)

# plt.figure(figsize=(8, 5))
# plt.hist(data_norm, bins=30, color='skyblue', edgecolor='black')
# plt.title("Гістограма нормального розподілу (μ=5, σ=2)")
# plt.xlabel("Значення")
# plt.ylabel("Частота")
# plt.show()

# 3
# hobbies = ['Читання книг', 'Data Science на Python', 'DIY електроніка', 'Силові тренування', 'Відеоігри']
# shares = [30, 25, 20, 15, 10]
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']

# plt.figure(figsize=(7, 7))
# plt.pie(shares, labels=hobbies, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
# plt.title("Розподіл часу на хобі")
# plt.show()

# 4
# apples = np.random.normal(loc=150, scale=15, size=100)
# bananas = np.random.normal(loc=120, scale=10, size=100)
# oranges = np.random.normal(loc=200, scale=20, size=100)
# pears = np.random.normal(loc=140, scale=12, size=100)

# fruit_data = [apples, bananas, oranges, pears]
# fruit_labels = ['Яблука', 'Банани', 'Апельсини', 'Груші']

# plt.figure(figsize=(8, 5))
# plt.boxplot(fruit_data, tick_labels=fruit_labels, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
# plt.title("Розподіл маси 4 типів фруктів")
# plt.ylabel("Маса (грами)")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# 5
# x5 = np.random.uniform(0, 1, 100)
# y5 = np.random.uniform(0, 1, 100)

# plt.figure(figsize=(8, 5))
# plt.scatter(x5, y5, color='green', alpha=0.6, edgecolors='darkgreen')
# plt.title("Точкова діаграма рівномірного розподілу")
# plt.xlabel("Ось X")
# plt.ylabel("Ось Y")
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.show()

# 6
# x6 = np.linspace(-2*np.pi, 2*np.pi, 500)
# y_sin = np.sin(x6)
# y_cos = np.cos(x6)
# y_sum = np.sin(x6) + np.cos(x6)

# plt.figure(figsize=(10, 6))
# plt.plot(x6, y_sin, color='blue', label='f(x) = sin(x)')
# plt.plot(x6, y_cos, color='red', label='g(x) = cos(x)', linestyle='--')
# plt.plot(x6, y_sum, color='purple', label='h(x) = sin(x) + cos(x)', linewidth=2)

# plt.title("Порівняння тригонометричних функцій")
# plt.xlabel("Значення X")
# plt.ylabel("Значення Y")
# plt.grid(True)
# plt.legend()
# plt.show()