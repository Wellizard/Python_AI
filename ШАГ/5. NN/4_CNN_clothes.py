import os
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

print("Завантаження датасету...")
path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Шлях до файлів датасету:", path)

train_path = os.path.join(path, "fashion-mnist_train.csv")
test_path = os.path.join(path, "fashion-mnist_test.csv")

print("Зчитування CSV файлів...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Нормалізація
X_train = X_train / 255.0
X_test = X_test / 255.0

# Зміна форми (Reshape) даних для CNN
# З (кількість, 784) на (кількість, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Створення CNN моделі
model = Sequential([
    Input(shape=(28, 28, 1)),
    
    # Перший згортковий шар
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Другий згортковий шар
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Перетворення двовимірних карт ознак у плоский вектор
    Flatten(),
    
    # Повнозв'язний шар
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Вихідний шар
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nПочаток тренування")
epochs = 20
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    validation_data=(X_test, y_test),
                    batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочність на тестових даних: {test_acc:.4f}')

# Побудова графіків
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Тренувальна точність (Train Acc)')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність (Val Acc)')
plt.title('Точність моделі (Accuracy)')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Тренувальна похибка (Train Loss)')
plt.plot(history.history['val_loss'], label='Валідаційна похибка (Val Loss)')
plt.title('Похибка моделі (Loss)')
plt.xlabel('Епохи')
plt.ylabel('Похибка (Crossentropy)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()