import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import speech_recognition as sr
from feedback import update_feedback, get_last_n_feedback, get_feedback, get_feedback_count

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

# Function to perform sentiment analysis
def perform_sentiment_analysis(text_input, model, vectorizer):
    # Preprocess the input text
    X = preprocess_text(text_input, vectorizer)

    # Predict sentiment
    sentiment = predict_sentiment(model, X)

    # Map sentiment labels to human-readable names
    sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    sentiment_label = sentiment_mapping.get(sentiment[0], 'Unknown')

    # Display sentiment result
    st.write("Sentiment:", sentiment_label)
    st.write("")

    # Save feedback
    update_feedback(text_input, sentiment[0])

    # Display line plot for sentiment means and density plot in parallel
    st.header("Sentiment Analysis")
    df_means = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Mean': [1.0 if sentiment_label == 'Positive' else 0.0,
                 1.0 if sentiment_label == 'Negative' else 0.0,
                 1.0 if sentiment_label == 'Neutral' else 0.0]
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Sentiment Means over Time
    sns.lineplot(data=df_means, x='Sentiment', y='Mean', marker='o', ax=ax1)
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Mean')
    ax1.set_title("Sentiment Means over Time")

    # Density Plot
    sns.barplot(data=df_means, x='Sentiment', y='Mean', palette='Set2', ax=ax2)
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Density')
    ax2.set_title("Density Plot")

    # Adjust layout and display
    plt.tight_layout()
    st.pyplot(fig)

# Function to handle voice input
def handle_voice_input():
    r = sr.Recognizer()
    mic_text = ""

    st.write("Click below to start speaking:")
    if st.button("Start Microphone"):
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            st.write("Listening... Speak command:")
            audio = r.listen(source)

        try:
            mic_text = r.recognize_google(audio)
            st.write(f"Command: {mic_text}")
        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

    return mic_text

# Main function
def main():
    st.title("Sentiment Analysis App")

    # Load pre-trained model and vectorizer
    model_dir = 'model'  # Change this to your model directory
    model, vectorizer = load_model_and_vectorizer(model_dir)

    # Sidebar with Model and Dataset Details
    st.sidebar.header("Model and Dataset Details")
    st.sidebar.write("Model and Vectorizer Size: TBD bytes")
    st.sidebar.write("Total Dataset Size: 400000")
    st.sidebar.write("Training Dataset Size: 320000")
    st.sidebar.write("Training Data Ratio: 80.00%")

    # Display model performance metrics and confusion matrix
    st.header("Model Performance")
    
    # Load dataset for accuracy calculation (just for demo purposes)
    df = pd.read_csv('data/Emotions.csv')
    X_train = vectorizer.transform(df['text'])
    y_train = df['label']
    y_pred = model.predict(X_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_pred) * 100
    st.metric(label="Model Accuracy", value=f"{accuracy:.2f}%")

    # Display classification report
    st.write("Classification Report")
    class_report = classification_report(y_train, y_pred)
    st.text(class_report)

    # Display confusion matrix
    st.write("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    # Display dataset distribution graph
    st.header("Dataset Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='label', palette='Set2', ax=ax1)
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # User input for sentiment analysis
    st.header("Analyze Sentiment")
    text_input = st.text_area("Enter text to analyze sentiment:", "")

    if text_input:
        perform_sentiment_analysis(text_input, model, vectorizer)

    # Voice command handling
    st.header("Voice Commands")
    mic_text = handle_voice_input()

    if mic_text:
        if "show last 5 feedback" in mic_text.lower():
            st.header("Last 5 Feedback Entries")
            last_n_feedback = get_last_n_feedback(5)
            st.write(last_n_feedback)
        elif "show positive feedback" in mic_text.lower():
            st.header("Positive Feedback Entries")
            positive_feedback = get_feedback(1)
            st.write(positive_feedback)
        elif "show negative feedback" in mic_text.lower():
            st.header("Negative Feedback Entries")
            negative_feedback = get_feedback(0)
            st.write(negative_feedback)
        elif "show neutral feedback" in mic_text.lower():
            st.header("Neutral Feedback Entries")
            neutral_feedback = get_feedback(2)
            st.write(neutral_feedback)
        else:
            st.write("Command not recognized. Try 'show last 5 feedback', 'show positive feedback', 'show negative feedback', or 'show neutral feedback'.")

if __name__ == "__main__":
    main()
