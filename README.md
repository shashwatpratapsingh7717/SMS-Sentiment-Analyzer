📱 SMS Sentiment AI: Advanced Sentiment Analyzer
An end-to-end Deep Learning project that classifies SMS messages into Positive, Negative, or Neutral sentiments using a Bidirectional LSTM neural network and Streamlit.

🚀 Overview
Traditional sentiment analyzers often struggle with context (e.g., "not happy"). This project uses a Bidirectional Long Short-Term Memory (LSTM) network to process text sequences in both directions, capturing deep contextual relationships between words.

Key Features:
Real-time Prediction: Instant sentiment analysis via a web interface.

Context Awareness: Uses Bidirectional LSTM to understand negations and complex sentence structures.

Data Visualization: Displays confidence scores for each sentiment category using dynamic bar charts.

Balanced Training: Implemented class weights to handle data imbalance issues.

🛠️ Tech Stack
Language: Python 3.x

Deep Learning: TensorFlow, Keras (LSTM)

Frontend: Streamlit

Data Handling: Pandas, NumPy

NLP Processing: Scikit-learn (Label Encoding), Keras Tokenizer

📂 Project Structure
Plaintext
├── app.py                # Streamlit Web Application
├── preprocess.py         # Data cleaning and label mapping
├── train_model.py        # Model architecture and training script
├── check_files.py        # Diagnostic script for model assets
├── sentiment_model.h5    # Trained LSTM Model
├── tokenizer.pkl         # Saved Word Tokenizer
├── label_encoder.pkl     # Saved Sentiment Label Encoder
└── requirements.txt      # List of dependencies
⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/YourUsername/SMS-Sentiment-AI.git
cd SMS-Sentiment-AI
Create and activate a virtual environment:

Bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # For Windows
Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py
🧠 Model Architecture
The model consists of:

Embedding Layer: Converts word indices into dense vectors.

Bidirectional LSTM Layer: 64 units with dropout for sequence learning.

Dense Layers: Fully connected layers with ReLU activation.

Softmax Output: Provides probability distribution across 3 classes.

📊 Results
The model achieves high accuracy by training on a diverse set of social media sentiments. By applying class_weights, we successfully reduced the false-neutral bias for negative messages.
