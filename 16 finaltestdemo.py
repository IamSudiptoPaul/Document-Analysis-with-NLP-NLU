import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Force CPU only to avoid Mac GPU "Metal" hangs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.clear_session()

# Load assets
df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/test-dataset1.csv').dropna(subset=['post'])
mlp = load_model('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_mlp_model.keras')
cnn = load_model('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_dl_model.keras')
vec_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_tfidf.pkl')
vec_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_bow.pkl')
lda = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_bow_model.pkl')
clusters = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/semantic_clusters.pkl')

TOPICS = {0:"Health", 1:"Politics", 2:"Legal", 3:"Economy", 4:"Social"}
results = []

print("\n--- Starting Rapid Inference (Top 10) ---")

# Process 10 samples to be safe for the demo
for i in range(10):
    text = df.iloc[i]['post']
    x_tfidf = vec_tfidf.transform([text]).toarray()
    
    # verbose=1 shows a progress bar so you know it's not stuck
    p_mlp = mlp.predict(x_tfidf, verbose=0)[0][0]
    p_cnn = cnn.predict(x_tfidf.reshape(1, 5000, 1), verbose=0)[0][0]
    
    topic_id = np.argmax(lda.transform(vec_bow.transform([text])))
    sem_id = clusters.predict(x_tfidf)[0]

    results.append({'MLP': p_mlp, 'CNN': p_cnn, 'Topic': TOPICS.get(topic_id)})
    print(f"Row {i+1}/10 processed...")

# --- Quick Plot ---
res_df = pd.DataFrame(results)
res_df[['MLP', 'CNN']].plot(kind='bar')
plt.title('Model Confidence Comparison')
plt.savefig('quick_demo.png')
print("\nSuccess! Results printed and 'quick_demo.png' created.")