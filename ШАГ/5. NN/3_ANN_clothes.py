import os
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
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

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential([
    Input(shape=(784,)),
    
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    
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