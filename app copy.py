# # import streamlit as st
# # import joblib
# # import pandas as pd
# # from sklearn.metrics import ConfusionMatrixDisplay
# # import matplotlib.pyplot as plt
# # from feedback import update_model

# # # Load the trained model and vectorizer
# # model = joblib.load('model/best_model.pkl')
# # vectorizer = joblib.load('model/vectorizer.pkl')

# # st.title("Sentiment Analysis App")

# # # Text input for user
# # user_input = st.text_area("Enter the text for sentiment analysis:")

# # if st.button("Analyze Sentiment"):
# #     if user_input:
# #         # Transform the user input text
# #         input_vector = vectorizer.transform([user_input])
        
# #         # Predict the sentiment
# #         prediction = model.predict(input_vector)[0]
        
# #         # Display the prediction
# #         if prediction == 1:
# #             st.success("The sentiment is Positive.")
# #         elif prediction == 0:
# #             st.warning("The sentiment is Neutral.")
# #         else:
# #             st.error("The sentiment is Negative.")
        
# #         # Display feedback options
# #         st.write("Was the prediction correct?")
# #         if st.button("Yes"):
# #             update_model(user_input, prediction)
# #             st.success("Thank you for your feedback!")
# #         if st.button("No"):
# #             st.write("Please select the correct sentiment:")
# #             feedback_label = st.radio("", ["Positive", "Neutral", "Negative"])
# #             feedback_value = 1 if feedback_label == "Positive" else (0 if feedback_label == "Neutral" else -1)
# #             update_model(user_input, feedback_value)
# #             st.success("Thank you for your feedback!")
# #     else:
# #         st.error("Please enter some text for analysis.")

# # st.write("## Model Evaluation")

# # # Load test data
# # test_data = pd.read_csv('data/Emotions.csv')
# # corpus = test_data['text']
# # y_test = test_data['label'].astype(int)

# # # Preprocess the test data
# # X_test = vectorizer.transform(corpus)

# # # Evaluate the model
# # y_pred = model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # precision = precision_score(y_test, y_pred, average='binary')
# # recall = recall_score(y_test, y_pred, average='binary')
# # conf_matrix = confusion_matrix(y_test, y_pred)
# # classification_rep = classification_report(y_test, y_pred)

# # st.write(f"Accuracy: {accuracy:.2f}")
# # st.write(f"Precision: {precision:.2f}")
# # st.write(f"Recall: {recall:.2f}")
# # st.write("Confusion Matrix:")
# # st.text(conf_matrix)
# # st.text("Classification Report:")
# # st.text(classification_rep)

# # # Plot Confusion Matrix
# # fig, ax = plt.subplots()
# # ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
# # st.pyplot(fig)

# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the trained model and vectorizer
# model = joblib.load("model/best_model.pkl")
# vectorizer = joblib.load("model/vectorizer.pkl")

# # Function to predict sentiment
# def predict_sentiment(text):
#     text_vectorized = vectorizer.transform([text])
#     prediction = model.predict(text_vectorized)
#     return prediction[0]

# # Streamlit App
# def main():
#     st.title("Sentiment Analysis App")

#     # Input text box for user input
#     user_input = st.text_input("Enter text:")

#     # Button to trigger prediction
#     if st.button("Predict"):
#         if user_input:
#             prediction = predict_sentiment(user_input)
#             st.write(f"Predicted Sentiment: {prediction}")

#             # Plotting the distribution of sentiment predictions
#             df = pd.DataFrame({"Sentiment": ["Negative", "Neutral", "Positive"], "Count": [0, 0, 0]})
#             df.loc[df['Sentiment'] == prediction, 'Count'] += 1

#             sns.set_style("whitegrid")
#             plt.figure(figsize=(8, 6))
#             sns.barplot(x="Sentiment", y="Count", data=df)
#             plt.title("Distribution of Sentiment Predictions")
#             plt.xlabel("Sentiment")
#             plt.ylabel("Count")
#             st.pyplot(plt)

#         else:
#             st.write("Please enter some text to predict.")

# if __name__ == "__main__":
#     main()

# # ----------------------3rd code revised---------------------------
# import streamlit as st
# import pandas as pd
# import joblib

# # Load the dataset
# data = pd.read_csv('data/Emotions.csv')

# # Load the trained model and vectorizer
# model = joblib.load('model/best_model.pkl')
# vectorizer = joblib.load('model/vectorizer.pkl')

# # Sidebar with dataset details
# st.sidebar.title("Dataset Details")
# st.sidebar.subheader("Number of Entries")
# st.sidebar.write(len(data))
# st.sidebar.subheader("Data Split Details")
# st.sidebar.write("Train:Test = 80:20")

# # Display model accuracy
# st.title("Sentiment Analysis App")
# st.subheader("Model Accuracy")
# st.write("Accuracy: {:.2f}%".format(accuracy))

# # User input for sentiment analysis
# user_input = st.text_input("Enter text to analyze sentiment:")
# if user_input:
#     # Vectorize the user input
#     user_input_vect = vectorizer.transform([user_input])

#     # Predict sentiment
#     prediction = model.predict(user_input_vect)[0]

#     # Display sentiment
#     st.write("Sentiment:")
#     if prediction == 0:
#         st.write("Negative")
#     elif prediction == 1:
#         st.write("Neutral")
#     elif prediction == 2:
#         st.write("Mixed")
#     elif prediction == 3:
#         st.write("Uncertain")
#     elif prediction == 4:
#         st.write("Positive")

# ------------------4> this code is working------------------------

# import streamlit as st
# import pandas as pd
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the pre-trained model and vectorizer
# def load_model_and_vectorizer(model_dir):
#     model_path = os.path.join(model_dir, 'best_model.pkl')
#     vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
#     model = joblib.load(model_path)
#     vectorizer = joblib.load(vectorizer_path)
#     return model, vectorizer

# # Process the input text
# def preprocess_text(text, vectorizer):
#     return vectorizer.transform([text])

# # Predict sentiment
# def predict_sentiment(model, X):
#     return model.predict(X)

# # Main function
# def main():
#     st.title("Sentiment Analysis App")
    
#     # Load pre-trained model and vectorizer
#     model_dir = 'model'  # Change this to your model directory
#     model, vectorizer = load_model_and_vectorizer(model_dir)

#     # Display data size
#     data_size = os.path.getsize('model/best_model.pkl') + os.path.getsize('model/vectorizer.pkl')
#     st.write(f"Data Size: {data_size} bytes")

#     # Get user input
#     text_input = st.text_area("Enter text to analyze sentiment:", "")

#     if text_input:
#         # Preprocess the input text
#         X = preprocess_text(text_input, vectorizer)

#         # Predict sentiment
#         sentiment = predict_sentiment(model, X)

#         # Map sentiment labels to human-readable names
#         sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
#         sentiment_label = sentiment_mapping.get(sentiment[0], 'Unknown')

#         # Display sentiment result
#         st.write("Sentiment:", sentiment_label)

#         # Calculate sentiment means
#         positive_mean = 0.0
#         negative_mean = 0.0
#         neutral_mean = 0.0
#         total = 1

#         if sentiment_label == 'Positive':
#             positive_mean += 1
#         elif sentiment_label == 'Negative':
#             negative_mean += 1
#         elif sentiment_label == 'Neutral':
#             neutral_mean += 1
#         total += 1

#         positive_mean /= total
#         negative_mean /= total
#         neutral_mean /= total

#         st.write("Positive Mean:", positive_mean)
#         st.write("Negative Mean:", negative_mean)
#         st.write("Neutral Mean:", neutral_mean)

#         # Display pie chart
#         labels = ['Positive', 'Negative', 'Neutral']
#         sizes = [positive_mean, negative_mean, neutral_mean]
#         colors = ['#ff9999','#66b3ff','#99ff99']
#         fig1, ax1 = plt.subplots()
#         ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#         ax1.axis('equal')  
#         st.pyplot(fig1)

# if __name__ == "__main__":
#     main()
# -----------------------------------end-----------------------------------------------
# -----------------------------------5th start-----------------------------------------------
# import streamlit as st
# import pandas as pd
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the pre-trained model and vectorizer
# def load_model_and_vectorizer(model_dir):
#     model_path = os.path.join(model_dir, 'best_model.pkl')
#     vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
#     model = joblib.load(model_path)
#     vectorizer = joblib.load(vectorizer_path)
#     return model, vectorizer

# # Process the input text
# def preprocess_text(text, vectorizer):
#     return vectorizer.transform([text])

# # Predict sentiment
# def predict_sentiment(model, X):
#     return model.predict(X)

# # Main function
# def main():
#     st.title("Sentiment Analysis App")
    
#     # Load pre-trained model and vectorizer
#     model_dir = 'model'  # Change this to your model directory
#     model, vectorizer = load_model_and_vectorizer(model_dir)

#     # Display data size
#     data_size = os.path.getsize('model/best_model.pkl') + os.path.getsize('model/vectorizer.pkl')
#     st.write(f"Data Size: {data_size} bytes")

#     # Display data ratio used for training
#     total_data_size = 400000  # Update this to your total dataset size
#     train_data_size = 320000  # Update this to your training dataset size
#     st.write(f"Total Dataset Size: {total_data_size}")
#     st.write(f"Training Dataset Size: {train_data_size}")
#     st.write(f"Training Data Ratio: {train_data_size/total_data_size*100:.2f}%")

#     # Get user input
#     text_input = st.text_area("Enter text to analyze sentiment:", "")

#     if text_input:
#         # Preprocess the input text
#         X = preprocess_text(text_input, vectorizer)

#         # Predict sentiment
#         sentiment = predict_sentiment(model, X)

#         # Map sentiment labels to human-readable names
#         sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
#         sentiment_label = sentiment_mapping.get(sentiment[0], 'Unknown')

#         # Display sentiment result
#         st.write("Sentiment:", sentiment_label)

#         # Calculate accuracy
#         accuracy = 85.0  # Update this with your actual accuracy
#         st.write("Accuracy:", accuracy)

#         # Calculate sentiment means
#         positive_mean = 0.0
#         negative_mean = 0.0
#         neutral_mean = 0.0
#         total = 1

#         if sentiment_label == 'Positive':
#             positive_mean += 1
#         elif sentiment_label == 'Negative':
#             negative_mean += 1
#         elif sentiment_label == 'Neutral':
#             neutral_mean += 1
#         total += 1

#         positive_mean /= total
#         negative_mean /= total
#         neutral_mean /= total

#         st.write("Positive Mean:", positive_mean)
#         st.write("Negative Mean:", negative_mean)
#         st.write("Neutral Mean:", neutral_mean)

#         # Display bar graph
#         fig, ax = plt.subplots()
#         ax.bar(['Positive', 'Negative', 'Neutral'], [positive_mean, negative_mean, neutral_mean])
#         ax.set_ylabel('Mean')
#         ax.set_title('Sentiment Means')
#         st.pyplot(fig)

#         # Display pie chart
#         labels = ['Positive', 'Negative', 'Neutral']
#         sizes = [positive_mean, negative_mean, neutral_mean]
#         colors = ['#ff9999','#66b3ff','#99ff99']
#         fig1, ax1 = plt.subplots()
#         ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#         ax1.axis('equal')  
#         st.pyplot(fig1)

# if __name__ == "__main__":
#     main()

# # -----------------------------------5th ended-----------------------------------------------
# # -----------------------------------6th Start > super working-----------------------------------------------
# import streamlit as st
# import pandas as pd
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the pre-trained model and vectorizer
# def load_model_and_vectorizer(model_dir):
#     model_path = os.path.join(model_dir, 'best_model.pkl')
#     vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
#     model = joblib.load(model_path)
#     vectorizer = joblib.load(vectorizer_path)
#     return model, vectorizer

# # Process the input text
# def preprocess_text(text, vectorizer):
#     return vectorizer.transform([text])

# # Predict sentiment
# def predict_sentiment(model, X):
#     return model.predict(X)

# # Main function
# def main():
#     st.title("Sentiment Analysis App")
    
#     # Load pre-trained model and vectorizer
#     model_dir = 'model'  # Change this to your model directory
#     model, vectorizer = load_model_and_vectorizer(model_dir)

#     # Display data size
#     data_size = os.path.getsize('model/best_model.pkl') + os.path.getsize('model/vectorizer.pkl')
#     st.write(f"Data Size: {data_size} bytes")

#     # Display data ratio used for training
#     total_data_size = 400000  # Update this to your total dataset size
#     train_data_size = 320000  # Update this to your training dataset size
#     st.write(f"Total Dataset Size: {total_data_size}")
#     st.write(f"Training Dataset Size: {train_data_size}")
#     st.write(f"Training Data Ratio: {train_data_size/total_data_size*100:.2f}%")

#     # Get user input
#     text_input = st.text_area("Enter text to analyze sentiment:", "")

#     if text_input:
#         # Preprocess the input text
#         X = preprocess_text(text_input, vectorizer)

#         # Predict sentiment
#         sentiment = predict_sentiment(model, X)

#         # Map sentiment labels to human-readable names
#         sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
#         sentiment_label = sentiment_mapping.get(sentiment[0], 'Unknown')

#         # Display sentiment result
#         st.write("Sentiment:", sentiment_label)

#         # Calculate accuracy
#         accuracy = 85.0  # Update this with your actual accuracy
#         st.write("Accuracy:", accuracy)

#         # Calculate sentiment means
#         positive_mean = 0.0
#         negative_mean = 0.0
#         neutral_mean = 0.0
#         total = 1

#         if sentiment_label == 'Positive':
#             positive_mean += 1
#         elif sentiment_label == 'Negative':
#             negative_mean += 1
#         elif sentiment_label == 'Neutral':
#             neutral_mean += 1
#         total += 1

#         positive_mean /= total
#         negative_mean /= total
#         neutral_mean /= total

#         st.write("Positive Mean:", positive_mean)
#         st.write("Negative Mean:", negative_mean)
#         st.write("Neutral Mean:", neutral_mean)

#         # Display density plot
#         df = pd.DataFrame({'Sentiment': [positive_mean, negative_mean, neutral_mean]}, index=['Positive', 'Negative', 'Neutral'])
#         st.write("Density Plot:")
#         st.line_chart(df)

#         # Display pair plot
#         st.write("Pair Plot:")
#         sns.pairplot(df, diag_kind='kde')
#         st.pyplot()

# if __name__ == "__main__":
#     main()

# # -----------------------------------6th ended-----------------------------------------------

# -----------7 start---------------------------------
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the pre-trained model and vectorizer
def load_model_and_vectorizer(model_dir):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Process the input text
def preprocess_text(text, vectorizer):
    return vectorizer.transform([text])

# Predict sentiment
def predict_sentiment(model, X):
    return model.predict(X)

# Main function
def main():
    st.title("Sentiment Analysis App")

    # Load pre-trained model and vectorizer
    model_dir = 'model'  # Change this to your model directory
    model, vectorizer = load_model_and_vectorizer(model_dir)

    # Sidebar with Model and Dataset Details
    st.sidebar.header("Model and Dataset Details")
    st.sidebar.write("Model and Vectorizer Size: TBD bytes")  # Update this with actual size
    st.sidebar.write("Total Dataset Size: 400000")
    st.sidebar.write("Training Dataset Size: 320000")
    st.sidebar.write("Training Data Ratio: 80.00%")

    # Display data size and dataset distribution
    st.header("Model and Dataset Details")

    # Load dataset for accuracy calculation
    df = pd.read_csv('data/Emotions.csv')
    X_train = vectorizer.transform(df['text'])
    y_train = df['label']
    y_pred = model.predict(X_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_pred) * 100

    # Display model performance metrics and confusion matrix
    st.header("Model Performance")
    st.metric(label="Model Accuracy", value=f"{accuracy:.2f}%")
    st.write("Classification Report")
    class_report = classification_report(y_train, y_pred)
    st.text(class_report)

    # Display confusion matrix and dataset distribution in parallel
    st.header("Confusion Matrix and Dataset Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)

    with col2:
        st.write("Dataset Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='label', palette='Set2', ax=ax1)
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

    # User input for sentiment analysis
    st.header("Analyze Sentiment")
    text_input = st.text_area("Enter text to analyze sentiment:", "")

    if text_input:
        # Preprocess the input text
        X = preprocess_text(text_input, vectorizer)

        # Predict sentiment
        sentiment = predict_sentiment(model, X)

        # Map sentiment labels to human-readable names
        sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        sentiment_label = sentiment_mapping.get(sentiment[0], 'Unknown')

        # Display sentiment result
        st.write(f"Sentiment: **{sentiment_label}**")

        # Calculate sentiment means
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_counts[sentiment_label.lower()] += 1

        positive_mean = sentiment_counts['positive'] / sum(sentiment_counts.values())
        negative_mean = sentiment_counts['negative'] / sum(sentiment_counts.values())
        neutral_mean = sentiment_counts['neutral'] / sum(sentiment_counts.values())

        st.write("Sentiment Means")
        st.write(f"Positive Mean: **{positive_mean:.2f}**")
        st.write(f"Negative Mean: **{negative_mean:.2f}**")
        st.write(f"Neutral Mean: **{neutral_mean:.2f}**")

        # Display line plot for sentiment means and density plot in parallel
        st.header("Sentiment Analysis")
        col3, col4 = st.columns(2)

        with col3:
            st.write("Sentiment Means over Time")
            df_means = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                     'Mean': [positive_mean, negative_mean, neutral_mean]})
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=df_means, x='Sentiment', y='Mean', marker='o', ax=ax2)
            st.pyplot(fig2)

        with col4:
            st.write("Density Plot")
            df_density = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Density': [positive_mean, negative_mean, neutral_mean]
            })
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=df_density, x='Sentiment', y='Density', palette='Set2', ax=ax3)
            ax3.set_xlabel('Sentiment')
            ax3.set_ylabel('Density')
            st.pyplot(fig3)

if __name__ == "__main__":
    main()

# -----------7 start---------------------------------