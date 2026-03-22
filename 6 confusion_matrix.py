#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Confusion Matrix Visualization for given model and data

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def run_evaluation(model_path, data_path):
    with open(data_path, 'rb') as f:
        _, X_test, _, y_test = pickle.load(f)

    model = tf.keras.models.load_model(model_path)
    X_test_final = X_test.toarray()

    # checking if the model is MLP or DL
    if len(model.input_shape) == 3:
        X_test_final = X_test_final.reshape(X_test_final.shape[0], X_test_final.shape[1], 1)

    y_pred = (model.predict(X_test_final) > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Fake', 'Real']).plot(cmap='Blues')
    
    plt.title(f"Results: {model_path}")
    plt.show()

run_evaluation('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_mlp_model.keras', '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/processed_data.pkl')