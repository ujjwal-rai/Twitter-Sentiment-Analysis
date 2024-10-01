---

# Twitter Sentiment Analysis

## Overview

This repository contains a Python project for sentiment analysis of tweets using natural language processing (NLP) techniques. The goal is to classify tweets into four categories: positive, negative, neutral, and irrelevant. The project utilizes machine learning algorithms to achieve this classification and presents visualizations of the data.

## Contents

- `twitter_training.csv`: The dataset used for training the sentiment analysis model.
- `notebooks/`: Jupyter notebook containing the entire code and analysis.
- `vectorizer.pkl`: The trained TfidfVectorizer model, saved for later use.
- `model.pkl`: The trained Random Forest Classifier model, saved for later use.

## Installation

To run this project, you'll need to install the required Python packages. You can do this using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
```

## Usage

1. **Load the Data**: The data is loaded from a CSV file named `twitter_training.csv`. The dataset contains the following columns:
   - `S.No.`: Serial number of the tweet.
   - `Context`: Context or additional information about the tweet (not used in the analysis).
   - `target`: Sentiment classification (0: irrelevant, 1: negative, 2: neutral, 3: positive).
   - `text`: The tweet text.

2. **Data Preprocessing**: 
   - Remove null values and duplicates.
   - Perform text transformations (lowercasing, tokenization, removing special characters, stopwords, and stemming).

3. **Exploratory Data Analysis (EDA)**: Visualize the distribution of sentiments and the characteristics of the tweet text (e.g., number of characters, words, and sentences).

4. **Model Training**:
   - Vectorization: Convert the processed text into numerical format using TfidfVectorizer.
   - Train-test Split: Split the dataset into training and testing sets.
   - Classification: Use different algorithms (GaussianNB, MultinomialNB, BernoulliNB, RandomForestClassifier) for sentiment classification.

5. **Evaluation**: Print accuracy scores and confusion matrices for each model.

6. **Model Saving**: Save the trained TfidfVectorizer and model using pickle.

## Important Notes

- Ensure that you run this code in a Python environment with access to the required libraries.
- The dataset used in this project is not included in the repository. You will need to obtain it separately and place it in the root directory.

## Visualizations

The project includes visualizations of:
- Sentiment distribution using pie charts.
- Character, word, and sentence counts using histograms.
- Word clouds for each sentiment category.

## Acknowledgments

This project makes use of the following libraries:
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [WordCloud](https://github.com/amueller/word_cloud)

---
