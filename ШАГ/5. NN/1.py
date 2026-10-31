import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
np.random.seed(42)

X = np.random.rand(1000, 2)

y = (X[:, 0] + X[:, 1] >= 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
model = keras.Sequential([
    layers.Dense(4, activation="relu", input_shape=(2,)),
    
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# ==========================================
history = model.fit(X_train, y_train, 
                    epochs=50,
                    batch_size=16, 
                    validation_data=(X_test, y_test), 
                    verbose=1)

# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n---> Точність (Accuracy) на тестовій вибірці: {test_acc * 100:.2f}%\n")

# ==========================================
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Навчальна вибірка (Train)')
plt.plot(history.history['val_loss'], label='Тестова вибірка (Validation)')
plt.title('Зміна функції втрат (Loss)')
plt.xlabel('Епохи')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Навчальна вибірка (Train)')
plt.plot(history.history['val_accuracy'], label='Тестова вибірка (Validation)')
plt.title('Зміна точності (Accuracy)')
plt.xlabel('Епохи')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


X_new = np.array([
    [0.1, 0.2],
    [0.8, 0.9],
    [0.5, 0.4],
    [0.6, 0.5],
    [0.9, 0.05]
])

predictions = model.predict(X_new)

predicted_classes = (predictions >= 0.5).astype(int)

print("\n---> Результати передбачення для 5 нових точок:")
for i in range(len(X_new)):
    x1, x2 = X_new[i]
    real_sum = x1 + x2
    pred_class = predicted_classes[i][0]
    prob = predictions[i][0]
    print(f"Точка (x1={x1:4.2f}, x2={x2:4.2f}) | Сума={real_sum:4.2f} | Передбачений клас: {pred_class} (Ймовірність: {prob:.4f})")