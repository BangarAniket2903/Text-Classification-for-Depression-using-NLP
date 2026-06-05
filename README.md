🧠 Depression Detection using NLP (Text Classification)


📌 Overview

This project is a Natural Language Processing (NLP) based Depression Detection System that classifies user input text as Depressed or Not Depressed.

The goal is to analyze mental health-related text patterns using machine learning and build a simple, interactive web application using Streamlit.


🚀 Live Demo

👉 [https://your-streamlit-app-link.streamlit.app](https://text-classification-for-depression-using-nlp.streamlit.app/)


📊 Problem Statement

Mental health issues such as depression are often reflected in language patterns. This project aims to:

Detect depressive language in text
Provide a lightweight and interpretable ML solution


📁 Dataset
Source: Reddit-based mental health dataset
Columns:
text → user input text
target → 0 (Not Depressed), 1 (Depressed)


🧹 Data Preprocessing

The following preprocessing steps were applied:

Lowercasing text
Removing URLs and HTML tags
Removing special characters, numbers, and punctuation
Removing stopwords
Tokenization
Lemmatization (used during experimentation)
Final cleaned text used for TF-IDF vectorization


🔠 Feature Engineering
Technique: TF-IDF Vectorization
Features selected: 3000 most important words
Converted text into numerical feature vectors


🤖 Model Building & Experiments

A wide range of machine learning models were evaluated to identify the best performer.

📌 1. Naive Bayes Family

We implemented and tested:

Gaussian Naive Bayes
Multinomial Naive Bayes
Bernoulli Naive Bayes

These models served as strong baseline classifiers for text data.

📌 2. Traditional Machine Learning Models

We trained and compared multiple algorithms:

Logistic Regression
Linear Support Vector Classifier (LinearSVC)
Decision Tree
Random Forest
Extra Trees Classifier
AdaBoost
Gradient Boosting

📌 3. Ensemble Methods

To improve performance further, we implemented:

✅ Voting Classifier
Combined models:
Random Forest
Extra Trees
Logistic Regression

✅ Stacking Classifier
Base models:
SVC
Logistic Regression
Extra Trees

Meta model:
Random Forest


🏆 Final Model Selection

After extensive experimentation and evaluation on metrics such as:

Accuracy
Precision
Recall
F1 Score
✔ Final Selected Model:

🥇 Logistic Regression with TF-IDF (3000 features)


⚠️ Disclaimer

This project is for educational purposes only.
It is not a medical diagnosis tool and should not replace professional mental health advice.


👨‍💻 Author

Aniket Bangar

B.Tech AI & Data Science
Passionate about Data, Machine Learning, and GenAI
⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

