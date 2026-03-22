#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Deep Learning with Convolutional Neural Networks (CNN) for Text Classification

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# reshaping the data because CNNs expect a 3D input (Batch, Steps, Features)
X_train_dl = X_train.toarray().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_dl = X_test.toarray().reshape(X_test.shape[0], X_test.shape[1], 1)

# designing a Deep Learning architecture using Convolutional layers
model = models.Sequential([
    # Input layer for 5000 features
    layers.Input(shape=(5000, 1)),
    
    # I am using a Conv1D layer to find patterns between neighboring word weights
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Adding a Flatten layer to transition to the fully connected part
    layers.Flatten(),
    
    # prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# I am training the model and saving the history for the demo visuals
history = model.fit(X_train_dl, y_train, 
                    epochs=55, 
                    batch_size=16, 
                    validation_split=0.1, 
                    verbose=1)

# I am saving this as the best DL model
model.save('best_dl_model.keras')

# I am creating a visual for the demo to show training vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Deep Learning Training Progress')
plt.legend()
plt.show()