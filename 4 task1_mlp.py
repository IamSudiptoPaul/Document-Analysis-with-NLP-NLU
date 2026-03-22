#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Multilayer Perceptron (MLP) for Text Classification

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the TF-IDF processed data
with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Converting sparse to dense for TF
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Build the optimized MLP Architecture
# Applying Hyperparameter (Funnel shape) and (Dropout)
model = models.Sequential([
    layers.Input(shape=(5000,)),
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.3),                  # Regularization to prevent overfitting
    layers.Dense(64, activation='relu'),  
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compiling with optimized Learning Rate (Hyperparameter #2)
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define Early Stopping
# This prevents the model from wasting time on epochs if it stops improving
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# Training
history = model.fit(X_train_dense, y_train, 
                    epochs=100, 
                    batch_size=8, 
                    validation_split=0.2, 
                    callbacks=[early_stop],
                    verbose=1)

# Evaluation
test_loss, test_acc = model.evaluate(X_test_dense, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Visualization of Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy Plot
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

# Loss Plot
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.show()

model.save('best_mlp_model.keras')
print("Model saved successfully as best_mlp_model.keras")

predictions = (model.predict(X_test_dense) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, predictions))