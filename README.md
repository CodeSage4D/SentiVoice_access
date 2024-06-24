# SentiVoice - Sentiment Analysis with Voice Commands

Welcome to SentiVoice, a sentiment analysis application that leverages both text input and voice commands to provide insights into the sentiment of user-provided texts. The app uses a pre-trained model to classify sentiments and allows users to interact using voice commands for an enhanced experience.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Voice Commands](#voice-commands)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Sentiment Analysis**: Analyze the sentiment of user-provided text as Positive, Negative, or Neutral.
- **Voice Commands**: Use voice commands to interact with the application and fetch feedback.
- **Feedback Management**: Save and retrieve feedback based on sentiment labels.
- **Data Visualization**: Display various graphs to visualize sentiment distribution and model performance.

## Installation

1. **Clone the Repository**
    ```sh
    git clone https://github.com/CodeSage4D/SentiVoice_access.git
    cd SentiVoice_access
    ```

2. **Create a Virtual Environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Prepare the Model and Vectorizer**
    - Place the pre-trained model (`best_model.pkl`) and vectorizer (`vectorizer.pkl`) in the `model` directory.

5. **Prepare the Dataset**
    - Place the dataset file (`Emotions.csv`) in the `data` directory.
    - Ensure the feedback CSV files (`positive_feedback.csv`, `negative_feedback.csv`, `neutral_feedback.csv`) are in the `data` directory.

## Usage

1. **Run the Application**
    ```sh
    streamlit run app.py
    ```

2. **Interact with the Application**
    - Open a browser and navigate to the provided local URL (e.g., `http://localhost:8501`).
    - Enter text in the provided text area to analyze its sentiment.
    - Use voice commands to interact with the application and fetch feedback.

## Voice Commands

### Available Commands:
- **Show Last 5 Feedback**: Displays the last 5 feedback entries.
- **Show First 5 Feedback**: Displays the first 5 feedback entries.
- **Show Positive Feedback**: Displays all positive feedback entries.
- **Show Negative Feedback**: Displays all negative feedback entries.
- **Show Neutral Feedback**: Displays all neutral feedback entries.

### Using Voice Commands:
1. Click on the "Start Microphone" button to activate the microphone.
2. Speak your command clearly into the microphone.
3. The application will process your command and display the requested information.

## File Structure
    SentiAnalysis/
    │
    ├── data/
    │   └── Emotions.csv
    │
    ├── model/
    │   ├── best_model.pkl
    │   └── vectorizer.pkl
    │
    │
    ├── feedback/
    │   ├── feedback\positive_feedback.csv
    │   └── feedback\neutral_feedback.csv
    │   └── feedback\negative_feedback.csv
    │   └── feedback\combined_feedback.csv
    │
    │
    ├── .gitignore
    ├── final.py
    ├── modelTrain.py
    ├── utils.py
    ├── app.py
    ├── feedback.py
    └── venv/

## Contributing
We welcome contributions to improve SentiVoice! If you have suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.

### Steps to Contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.


