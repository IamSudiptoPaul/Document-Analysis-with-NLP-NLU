#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Model Testing for new posts using the given best trained model path MLP or DL_CNN

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pickle
import tensorflow as tf
import numpy as np

# need the vectorizer to turn new words into the same numbers the model knows
with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# best trained model
model = tf.keras.models.load_model('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_dl_model.keras')

def predict_post(new_post_text):
    vectorized_text = vectorizer.transform([new_post_text]).toarray()
    
    # Reshape for the DL_CNN model (Batch, Features, 1)
    # If testing the MLP, can be skipped this reshape step
    input_data = vectorized_text.reshape(vectorized_text.shape[0], vectorized_text.shape[1], 1)
    
    prediction_prob = model.predict(input_data)[0][0]
    
    # 0.5 is threshold here
    label = "TRUE (Real News)" if prediction_prob > 0.5 else "FALSE (Fake News)"
    confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
    
    print(f"\nPost: {new_post_text}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2%}")

# demo run with a sample post
sample_post = "The government has not announced a total ban on all evictions starting tomorrow."
predict_post(sample_post)