# # # # import joblib
# # # # import pandas as pd

# # # # def update_model(feedback_text, feedback_label, model_path='model/best_model.pkl', vectorizer_path='model/vectorizer.pkl', data_path='data/feedback_data.csv'):
# # # #     # Load the existing model and vectorizer
# # # #     model = joblib.load(model_path)
# # # #     vectorizer = joblib.load(vectorizer_path)
    
# # # #     # Transform the feedback text using the vectorizer
# # # #     feedback_vector = vectorizer.transform([feedback_text])
    
# # # #     # Update the model (we'll use incremental learning here if applicable)
# # # #     # Otherwise, save the feedback to a file for later retraining
# # # #     feedback_data = pd.DataFrame({
# # # #         'text': [feedback_text],
# # # #         'label': [feedback_label]
# # # #     })
    
# # # #     feedback_data.to_csv(data_path, mode='a', header=False, index=False)
    
# # # #     print("Feedback saved. Retrain the model later with the new feedback data.")

# # # # # Example usage:
# # # # # update_model("This is a positive text", 1)

# # # # def collect_feedback():
# # # #     print("Please provide your feedback:")
# # # #     feedback = input()
# # # #     print("Thank you for your feedback!")

# # # #     # Save the feedback to a file or database
# # # #     save_feedback(feedback)

# # # # def save_feedback(feedback):
# # # #     # Write the feedback to a file or database
# # # #     with open("feedback.txt", "a") as file:
# # # #         file.write(feedback + "\n")

# # # # def main():
# # # #     print("Welcome to the Feedback Collection System!")
# # # #     while True:
# # # #         print("1. Provide Feedback")
# # # #         print("2. Exit")
# # # #         choice = input("Enter your choice: ")

# # # #         if choice == "1":
# # # #             collect_feedback()
# # # #         elif choice == "2":
# # # #             print("Exiting the program. Thank you!")
# # # #             break
# # # #         else:
# # # #             print("Invalid choice. Please try again.")

# # # # if __name__ == "__main__":
# # # #     main()

# # # # ---------------------------2 start----------------------------------
# # # # import joblib
# # # # import pandas as pd
# # # # import os

# # # # def update_model(feedback_text, feedback_label, model_path='model/best_model.pkl', vectorizer_path='model/vectorizer.pkl', data_path='data/feedback.csv'):
# # # #     # Load the existing model and vectorizer
# # # #     model = joblib.load(model_path)
# # # #     vectorizer = joblib.load(vectorizer_path)
    
# # # #     # Transform the feedback text using the vectorizer
# # # #     feedback_vector = vectorizer.transform([feedback_text])
    
# # # #     # Save the feedback to a single CSV file
# # # #     feedback_data = pd.DataFrame({
# # # #         'text': [feedback_text],
# # # #         'label': [feedback_label]
# # # #     })
    
# # # #     if not os.path.isfile(data_path):
# # # #         feedback_data.to_csv(data_path, index=False)
# # # #     else:
# # # #         feedback_data.to_csv(data_path, mode='a', header=False, index=False)
    
# # # #     print(f"Feedback saved to {data_path}. Retrain the model later with the new feedback data.")

# # # # # Example usage:
# # # # # update_model("This is a positive text", 'Positive')

# # # # def collect_feedback():
# # # #     print("Please provide your feedback:")
# # # #     feedback = input()
# # # #     print("Thank you for your feedback!")

# # # #     # Save the feedback to a file or database
# # # #     save_feedback(feedback)

# # # # def save_feedback(feedback):
# # # #     # Write the feedback to a file or database
# # # #     with open("feedback.txt", "a") as file:
# # # #         file.write(feedback + "\n")

# # # # def main():
# # # #     print("Welcome to the Feedback Collection System!")
# # # #     while True:
# # # #         print("1. Provide Feedback")
# # # #         print("2. Exit")
# # # #         choice = input("Enter your choice: ")

# # # #         if choice == "1":
# # # #             collect_feedback()
# # # #         elif choice == "2":
# # # #             print("Exiting the program. Thank you!")
# # # #             break
# # # #         else:
# # # #             print("Invalid choice. Please try again.")

# # # # if __name__ == "__main__":
# # # #     main()

# # # # ---------------------------2 end----------------------------------

# # ----------------------------------------------------------------------------------

# import pandas as pd
# import os

# # Directory to store feedback files
# FEEDBACK_DIR = 'feedback_files'

# # Ensure feedback directory exists
# if not os.path.exists(FEEDBACK_DIR):
#     os.makedirs(FEEDBACK_DIR)

# # File paths for each sentiment
# POSITIVE_FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'positive_feedback.csv')
# NEGATIVE_FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'negative_feedback.csv')
# NEUTRAL_FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'neutral_feedback.csv')

# # Function to update feedback
# def update_feedback(text, sentiment):
#     feedback_entry = pd.DataFrame({'text': [text], 'sentiment': [sentiment]})

#     if sentiment == 1:  # Positive
#         if os.path.exists(POSITIVE_FEEDBACK_FILE):
#             positive_feedback = pd.read_csv(POSITIVE_FEEDBACK_FILE)
#             positive_feedback = pd.concat([positive_feedback, feedback_entry], ignore_index=True)
#         else:
#             positive_feedback = feedback_entry
#         positive_feedback.to_csv(POSITIVE_FEEDBACK_FILE, index=False)
#     elif sentiment == 0:  # Negative
#         if os.path.exists(NEGATIVE_FEEDBACK_FILE):
#             negative_feedback = pd.read_csv(NEGATIVE_FEEDBACK_FILE)
#             negative_feedback = pd.concat([negative_feedback, feedback_entry], ignore_index=True)
#         else:
#             negative_feedback = feedback_entry
#         negative_feedback.to_csv(NEGATIVE_FEEDBACK_FILE, index=False)
#     elif sentiment == 2:  # Neutral
#         if os.path.exists(NEUTRAL_FEEDBACK_FILE):
#             neutral_feedback = pd.read_csv(NEUTRAL_FEEDBACK_FILE)
#             neutral_feedback = pd.concat([neutral_feedback, feedback_entry], ignore_index=True)
#         else:
#             neutral_feedback = feedback_entry
#         neutral_feedback.to_csv(NEUTRAL_FEEDBACK_FILE, index=False)

# # Function to get feedback by label
# def get_feedback_by_label(label, limit=None):
#     feedback_files = {
#         'positive': POSITIVE_FEEDBACK_FILE,
#         'negative': NEGATIVE_FEEDBACK_FILE,
#         'neutral': NEUTRAL_FEEDBACK_FILE,
#         'all': [POSITIVE_FEEDBACK_FILE, NEGATIVE_FEEDBACK_FILE, NEUTRAL_FEEDBACK_FILE]
#     }

#     if label == 'all':
#         feedback_data = pd.concat([pd.read_csv(f) for f in feedback_files['all'] if os.path.exists(f)], ignore_index=True)
#     else:
#         feedback_data = pd.read_csv(feedback_files[label]) if os.path.exists(feedback_files[label]) else pd.DataFrame()

#     if limit:
#         return feedback_data.tail(limit)
#     else:
#         return feedback_data
# ===================================================================
# This code for feedback take and update work very well---------------------------Final Feedback.py-----------------
import pandas as pd
import os

FEEDBACK_DIR = 'feedback'

# Ensure feedback directory exists
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def update_feedback(text, sentiment):
    sentiment_mapping = {0: 'negative_feedback.csv', 1: 'positive_feedback.csv', 2: 'neutral_feedback.csv'}
    filename = sentiment_mapping[sentiment]
    filepath = os.path.join(FEEDBACK_DIR, filename)

    df = pd.DataFrame([[text, sentiment]], columns=['feedback', 'sentiment'])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, mode='w', header=True, index=False)

    # Update combined feedback
    update_combined_feedback()

def update_combined_feedback():
    combined_path = os.path.join(FEEDBACK_DIR, 'combined_feedback.csv')
    df_list = []
    for filename in ['negative_feedback.csv', 'positive_feedback.csv', 'neutral_feedback.csv']:
        filepath = os.path.join(FEEDBACK_DIR, filename)
        if os.path.exists(filepath):
            df_list.append(pd.read_csv(filepath))
    combined_df = pd.concat(df_list)
    combined_df.to_csv(combined_path, index=False)

def get_last_n_feedback(n):
    combined_path = os.path.join(FEEDBACK_DIR, 'combined_feedback.csv')
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        return df.tail(n)
    return pd.DataFrame(columns=['feedback', 'sentiment'])

def get_feedback(sentiment):
    sentiment_mapping = {0: 'negative_feedback.csv', 1: 'positive_feedback.csv', 2: 'neutral_feedback.csv'}
    filename = sentiment_mapping[sentiment]
    filepath = os.path.join(FEEDBACK_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame(columns=['feedback', 'sentiment'])

def get_feedback_count():
    combined_path = os.path.join(FEEDBACK_DIR, 'combined_feedback.csv')
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        return df['sentiment'].value_counts().to_dict()
    return {0: 0, 1: 0, 2: 0}

# ------------------------------------------Final feedback.py end ------------------------------------------