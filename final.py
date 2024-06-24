# import pandas as pd
# import time
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# from modelTrain import train_model
# from utils import preprocess_data
# from tqdm import tqdm

# def main():
#     # Load the dataset
#     df = pd.read_csv('data/Emotions.csv')
#     corpus = df['text']  # Assuming the text column is named 'text'

#     # Preprocess the data
#     print("Preprocessing data...")
#     X, y, cv = preprocess_data(df, corpus)

#     # Split the data
#     print("Splitting data...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
#     total_size = len(y)
#     train_size = len(y_train)
#     test_size = len(y_test)

#     print(f"Total dataset size: {total_size}")
#     print(f"Training dataset size: {train_size}")
#     print(f"Testing dataset size: {test_size}")

#     # Ensure the model directory exists
#     os.makedirs('model', exist_ok=True)

#     # Train and evaluate the model
#     start_time = time.time()
#     print("Training the model...")

#     progress_bar = tqdm(total=100, position=0, leave=True)
#     progress = 0
#     while progress < 100:
#         time.sleep(0.1)  # Simulating the training process
#         progress += 1
#         progress_bar.update(1)
#         elapsed_time = time.time() - start_time
#         print(f"\r[{'=' * progress}{' ' * (100 - progress)}] {progress}% [{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}]", end='')
    
#     progress_bar.close()

#     best_model, best_params = train_model(X_train, y_train)

#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     print(f"\nBest Parameters: {best_params}")
#     print(f"Elapsed Time: {elapsed_time:.2f} seconds")

#     # Save the model and the vectorizer
#     print("Saving the model and vectorizer...")
#     joblib.dump(best_model, 'model/best_model.pkl')
#     joblib.dump(cv, 'model/vectorizer.pkl')
#     print("Model and vectorizer saved successfully!")

#     # Evaluate the model
#     print("Evaluating the model...")
#     y_pred = best_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='binary')
#     recall = recall_score(y_test, y_pred, average='binary')
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred)

#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print("Classification Report:")
#     print(classification_rep)

#     # Plot Confusion Matrix
#     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
#     disp.plot()
#     plt.show()

#     # Check if the model folder is empty
#     model_dir_empty = len(os.listdir('model')) == 0
#     if model_dir_empty:
#         print("Model folder is empty. There may be an issue with saving the model.")
#     else:
#         print("Model trained successfully.")

# if __name__ == "__main__":
#     main()

# ---------------------code run well------------------
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# from tqdm import tqdm
# import joblib
# import os

# # Load the dataset
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     return df

# # Preprocess data
# def preprocess_data(df, max_data=None):
#     start_time = time.time()
#     if max_data:
#         df = df.sample(n=max_data, random_state=42)
#     corpus = df['text']
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X = vectorizer.fit_transform(tqdm(corpus, desc="Vectorizing Data"))
#     y = df['label']
#     end_time = time.time()
#     preprocess_time = end_time - start_time
#     print("Preprocessing time:", preprocess_time, "seconds")
#     return X, y, vectorizer

# # Train the model
# def train_model(X, y):
#     start_time = time.time()
#     model = MultinomialNB()
#     model.fit(X, y)
#     end_time = time.time()
#     train_time = end_time - start_time
#     print("Training time:", train_time, "seconds")
#     return model

# # Evaluate the model
# def evaluate_model(model, X_test, y_test):
#     start_time = time.time()
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred)
#     end_time = time.time()
#     evaluation_time = end_time - start_time
#     print("Evaluation time:", evaluation_time, "seconds")
#     return accuracy, confusion_mat

# # Plot confusion matrix
# def plot_confusion_matrix(confusion_mat):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Save the model
# def save_model(model, vectorizer):
#     if not os.path.exists('model'):
#         os.makedirs('model')
#     joblib.dump(model, 'model/best_model.pkl')
#     joblib.dump(vectorizer, 'model/vectorizer.pkl')
#     print("Model saved successfully.")

# def main():
#     # Load data
#     file_path = 'data/Emotions.csv'  # Change this to your file path
#     df = load_data(file_path)

#     # Specify the maximum data size
#     max_data = None  # Set this to the desired maximum data size, or None to use all data

#     # Preprocess data
#     X, y, vectorizer = preprocess_data(df, max_data=max_data)

#     # Train the model
#     model = train_model(X, y)

#     # Evaluate the model
#     X_test = vectorizer.transform(df['text'])  # Assuming you want to evaluate on the full dataset
#     y_test = df['label']
#     accuracy, confusion_mat = evaluate_model(model, X_test, y_test)
#     print("Accuracy:", accuracy)
#     print("Confusion Matrix:")
#     print(confusion_mat)

#     # Plot confusion matrix
#     plot_confusion_matrix(confusion_mat)

#     # Save the model
#     save_model(model, vectorizer)

# if __name__ == "__main__":
#     main()

# ---------------------code run well end------------------
# ---------------------1 start------------------
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df, max_data=None):
    start_time = time.time()
    if max_data:
        df = df.sample(n=max_data, random_state=42)
    corpus = df['text']
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(tqdm(corpus, desc="Vectorizing Data"))
    y = df['label']
    end_time = time.time()
    preprocess_time = end_time - start_time
    print("Preprocessing time:", preprocess_time, "seconds")
    return X, y, vectorizer

# Train the model
def train_model(X, y):
    start_time = time.time()
    model = MultinomialNB()
    model.fit(X, y)
    end_time = time.time()
    train_time = end_time - start_time
    print("Training time:", train_time, "seconds")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    end_time = time.time()
    evaluation_time = end_time - start_time
    print("Evaluation time:", evaluation_time, "seconds")
    return accuracy, conf_matrix, class_report

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Save the model
def save_model(model, vectorizer):
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(model, 'model/best_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    print("Model saved successfully.")

def main():
    # Load data
    file_path = 'data/Emotions.csv'  # Change this to your file path
    df = load_data(file_path)

    # Specify the maximum data size
    max_data = None  # Set this to the desired maximum data size, or None to use all data

    # Preprocess data
    X, y, vectorizer = preprocess_data(df, max_data=max_data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Save the model
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()


# ---------------------1 code end------------------

