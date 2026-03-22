import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    print("Note: FlagEmbedding not installed. BGE-M3 will run in 'Offline Cluster Mode'.")

# 1. SETTINGS & TOPIC MAPPING
# These labels are based on your analysis of the cluster keywords
TOPIC_LABELS = {
    0: "Public Health (Vaccines/COVID-19)",
    1: "Political Discourse (Elections/Biden/Trump)",
    2: "Legal & Constitutional Rights",
    3: "Finance, Economy & Inflation",
    4: "Social Media & Digital Communication"
}

# 2. LOAD ALL ASSETS
print("Loading models and vectorizers...")
test_df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/test-dataset1.csv').dropna(subset=['post'])

mlp_model = load_model('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_mlp_model.keras')
cnn_model = load_model('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_dl_model.keras')
vec_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_tfidf.pkl')
vec_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_bow.pkl')
lda_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_bow_model.pkl')
lda_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_tfidf_model.pkl')
semantic_cluster_model = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/semantic_clusters.pkl')

# BGE-M3 Semantic Model (Optional: requires FlagEmbedding library)
try:
    # use_fp16=True optimizes performance on MacBook Air
    bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    HAS_BGE = True
except:
    HAS_BGE = False

def run_master_inference(idx):
    row = test_df.iloc[idx]
    text = row['post']
    ground_truth = "TRUE (Real)" if row['class_label'] == True else "FALSE (Fake)"

    print(f"\n{'='*80}")
    print(f"DEMO CASE #{idx+1} | EXPECTED: {ground_truth}")
    print(f"TEXT: {text[:150]}...")
    print(f"{'='*80}")

    # --- PART 1: CLASSIFICATION (MLP & CNN) ---
    x_tfidf = vec_tfidf.transform([text]).toarray()
    
    # MLP Inference
    mlp_prob = mlp_model.predict(x_tfidf, verbose=0)[0][0]
    mlp_res = "TRUE (Real)" if mlp_prob > 0.5 else "FALSE (Fake)"
    mlp_conf = mlp_prob if mlp_prob > 0.5 else (1 - mlp_prob)

    # CNN Inference (Reshaped for 1D-Convolution)
    x_cnn = x_tfidf.reshape(1, x_tfidf.shape[1], 1)
    cnn_prob = cnn_model.predict(x_cnn, verbose=0)[0][0]
    cnn_res = "TRUE (Real)" if cnn_prob > 0.5 else "FALSE (Fake)"
    cnn_conf = cnn_prob if cnn_prob > 0.5 else (1 - cnn_prob)

    print(f" [TASK 1: Classification]")
    print(f"  > MLP (TF-IDF):   {mlp_res} (Confidence: {mlp_conf:.2%})")
    print(f"  > CNN (Sequence): {cnn_res} (Confidence: {cnn_conf:.2%})")

    # --- PART 2: TOPIC MODELING (LDA) ---
    x_bow = vec_bow.transform([text])
    lda_id = np.argmax(lda_bow.transform(x_bow))
    
    print(f"\n [TASK 2: Statistical Topics]")
    print(f"  > LDA (BoW):      Topic {lda_id} - {TOPIC_LABELS.get(lda_id)}")

    # --- PART 3: SEMANTIC NLU (BGE-M3) ---
    print(f"\n [TASK 2: Semantic Discovery]")
    if HAS_BGE:
        # Live vector generation
        embedding = bge_model.encode([text])['dense_vecs'][0]
        # Predict cluster based on embedding
        sem_id = semantic_cluster_model.predict([embedding])[0]
        print(f"  > BGE-M3 Vector:  Generated 1024-dim 'Meaning' Embedding")
    else:
        # Proxy using TF-IDF for demo speed if model not loaded
        sem_id = semantic_cluster_model.predict(x_tfidf)[0]
    
    print(f"  > Semantic Group: {TOPIC_LABELS.get(sem_id)} (Context-Aware)")
    print(f"{'='*80}")

# Execute demo for 3 distinct samples
for i in [0, 5, 10]:
    run_master_inference(i)