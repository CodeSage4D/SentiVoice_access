# from sklearn.feature_extraction.text import CountVectorizer

# def preprocess_data(df, corpus):
#     # Update the column name here to match your dataset
#     y = df['label']  # Assuming the label column is named 'label'

#     # Initialize the CountVectorizer
#     cv = CountVectorizer()
#     X = cv.fit_transform(corpus)
    
#     return X, y, cv


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(df, corpus=None):
    """
    Preprocesses the data by splitting it into features and labels.

    Args:
    - df: pandas DataFrame containing the dataset
    - corpus: Optional corpus to be used for vectorization

    Returns:
    - X: Features (text data)
    - y: Labels
    - cv: Count Vectorizer if corpus is provided, otherwise None
    """
    # Assuming 'text' is your feature column and 'label' is your target column
    X = df['text']
    y = df['label']

    # If corpus is provided, use it for vectorization
    if corpus:
        cv = TfidfVectorizer(vocabulary=corpus)
        X = cv.fit_transform(X)
        return X, y, cv
    else:
        return X, y, None

def split_data(X, y, test_size=0.25, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
    - X: Features
    - y: Labels
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed used by the random number generator

    Returns:
    - X_train: Features for training
    - X_test: Features for testing
    - y_train: Labels for training
    - y_test: Labels for testing
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
