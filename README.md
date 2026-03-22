# Document-Analysis-with-NLP-NLU
This repository contains the implementation of advanced Natural Language Processing (NLP) and Artificial Neural Network (ANN) models designed for text document classification and topic discovery. The project focuses on analyzing a large-scale social media dataset to identify fake content and extract underlying thematic structures.

🛠️ Key Tasks

1. Identification of Fake Text Documents

Data Processing: Implementation of syntactic analysis and transformation of text into numerical feature vectors.

Shallow Learning: Design and training of a Multi-Layer Perceptron (MLP) to predict the truthfulness (class_label) of social media posts.

Deep Learning: Development of a Deep Learning (DL) architecture, utilizing specialized layers (such as convolutional or fully connected layers) to improve classification accuracy.

Optimization: Systematic hyperparameter tuning of three key variables to refine model performance against baseline results.

2. Topic Discovery with Natural Language Understanding (NLU)

Text Representation: Comparison of at least two representation strategies, including BoW, TF-IDF, LDA, or Word Embeddings.

Thematic Analysis: Automated discovery of underlying topics across text data and news headlines.

Evaluation: Analysis of the link between discovered topics, document content, and ground-truth labels.

📊 Dataset Details

The project utilizes the social-media.csv dataset, which contains over 89,000 posts linked to 947 news headlines.

Annotation: Labels were determined by matching news headline ground truth with majority votes from participants via Amazon Mechanical Turk.

Features: Includes id, news_headline, news_headline_ground_truth, post, majority_votes, and class_label.

💻 Technical Stack

Language: Python.

Libraries: PyTorch, TensorFlow, ScikitLearn, and standard NLP packages.

Techniques: Lemmatization, stemming, stop word removal, and syntactic parsing
