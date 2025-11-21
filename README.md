# Duplicate Question Detection

## Description
This project predicts whether two questions are duplicates using text similarity analysis. Two models were implemented:

1. **Logistic Regression (Baseline)** – TF-IDF features, class imbalance handled with `class_weight='balanced'`.
2. **Siamese LSTM** – Shared LSTM encoder for question pairs, class weights applied, dropout for regularization, and early stopping to prevent overfitting.

Key decisions:
- Text preprocessing: tokenization, lowercasing, removing special characters, stopword removal, lemmatization.
- Feature extraction: TF-IDF and Word2Vec embeddings (average of token vectors).
- Model tuning: experimented with architectures, activations, dropout, and optimizers to improve performance.

