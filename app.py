import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import re
import speech_recognition as sr

# Feedback File Path
FEEDBACK_FILE = "feedback/combined_feedback.csv"

# Load Hugging Face Sentiment Analysis
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_pipeline()

# Ensure feedback directory exists
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

# Initialize feedback storage
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["text", "sentiment", "score"]).to_csv(FEEDBACK_FILE, index=False)

# Save feedback
def save_feedback(feedback_text, sentiment, score):
    df = pd.read_csv(FEEDBACK_FILE)
    new_entry = pd.DataFrame({"text": [feedback_text], "sentiment": [sentiment], "score": [score]})
    updated_df = pd.concat([df, new_entry], ignore_index=True)
    updated_df.to_csv(FEEDBACK_FILE, index=False)

# Retrieve feedback
def get_feedback(sentiment_filter=None, count=5, position="last"):
    try:
        df = pd.read_csv(FEEDBACK_FILE)
        if sentiment_filter:
            df = df[df["sentiment"].str.upper() == sentiment_filter.upper()]
        if position == "first":
            df = df.head(count)
        else:
            df = df.tail(count)
        return df.to_dict("records") if not df.empty else []
    except Exception as e:
        st.error(f"Error retrieving feedback: {e}")
        return []

# Count feedback by sentiment
def count_feedback():
    try:
        df = pd.read_csv(FEEDBACK_FILE)
        counts = df["sentiment"].value_counts().to_dict()
        return {sent.upper(): counts.get(sent, 0) for sent in ["POSITIVE", "NEGATIVE", "NEUTRAL"]}
    except Exception as e:
        st.error(f"Error counting feedback: {e}")
        return {}

# Handle voice commands
def handle_voice_command(command):
    command = command.lower()
    feedback = []
    try:
        # Extract number and sentiment using regex
        number_match = re.search(r"(\d+)", command)
        count = int(number_match.group()) if number_match else 5
        sentiment = None

        if "positive" in command:
            sentiment = "POSITIVE"
        elif "negative" in command:
            sentiment = "NEGATIVE"
        elif "neutral" in command:
            sentiment = "NEUTRAL"

        # Determine position (first/last)
        position = "first" if "first" in command else "last"

        # Process command
        if "show" in command and "feedback" in command:
            feedback = get_feedback(sentiment_filter=sentiment, count=count, position=position)
        elif "count feedback" in command:
            feedback = count_feedback()
        elif "all feedback" in command:
            feedback = get_feedback(sentiment_filter=None, count=1000)
    except Exception as e:
        st.error(f"Error processing command: {e}")
    return feedback

# Streamlit App
def main():
    st.title("Feedback Sentiment Tracker with Voice Commands")
    st.info("Supported Commands:\n\n"
            "1. 'Show last 5 positive feedback.'\n"
            "2. 'Show first 3 negative feedback.'\n"
            "3. 'Count feedback.'\n"
            "4. 'Show all feedback.'\n"
            "5. 'Show last 2 neutral feedback.'")

    # Section 1: Input Feedback
    st.subheader("Submit Your Feedback")
    feedback_text = st.text_area("Enter your feedback:", height=100)
    if st.button("Analyze and Save Feedback"):
        if feedback_text.strip():
            # Analyze Sentiment
            analysis = sentiment_analyzer(feedback_text)[0]
            sentiment = analysis["label"]
            score = round(analysis["score"] * 100, 2)

            # Save Feedback
            save_feedback(feedback_text, sentiment, score)
            st.success(f"Feedback saved! Sentiment: **{sentiment}** (Score: {score}%)")
        else:
            st.error("Please enter valid feedback!")

    # Section 2: Command Testing
    st.subheader("Test Commands")
    commands = [
        "Show last 5 positive feedback",
        "Show first 3 negative feedback",
        "Show last 2 neutral feedback",
        "Count feedback",
        "Show all feedback"
    ]
    selected_command = st.selectbox("Select a command to test:", commands)
    if st.button("Execute Command"):
        feedback = handle_voice_command(selected_command)
        if isinstance(feedback, list):
            if feedback:
                st.write("Retrieved feedback entries:")
                for idx, fb in enumerate(feedback, start=1):
                    st.write(f"**{idx}. {fb['text']}** - Sentiment: {fb['sentiment']} (Score: {fb['score']}%)")
            else:
                st.warning("No feedback entries found!")
        elif isinstance(feedback, dict):
            st.write("Feedback Count by Sentiment:")
            for sentiment, count in feedback.items():
                st.write(f"**{sentiment}: {count} entries**")
        else:
            st.error("No feedback available for the given command.")

    # Section 3: Voice Commands
    st.subheader("Voice Command Mode")
    st.text("Click below to start speaking:")
    if st.button("Start Voice Command"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak your command:")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                st.write(f"Command: {command}")

                # Handle Command
                feedback = handle_voice_command(command)
                if isinstance(feedback, list):
                    if feedback:
                        st.write("Retrieved feedback entries:")
                        for idx, fb in enumerate(feedback, start=1):
                            st.write(f"**{idx}. {fb['text']}** - Sentiment: {fb['sentiment']} (Score: {fb['score']}%)")
                    else:
                        st.warning("No feedback entries found!")
                elif isinstance(feedback, dict):
                    st.write("Feedback Count by Sentiment:")
                    for sentiment, count in feedback.items():
                        st.write(f"**{sentiment}: {count} entries**")
                else:
                    st.error("No feedback available for the given command.")
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please try again.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
