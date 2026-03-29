import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(page_title="SMS Sentiment AI", page_icon="🧪")

# Load assets with caching to save memory
@st.cache_resource
def load_all_files():
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le

# Initialize
try:
    model, tokenizer, le = load_all_files()
except Exception as e:
    st.error("Could not load model files. Please run the training scripts first.")
    st.stop()

# UI Layout
st.title("🧪 SMS Sentiment Analyzer")
st.markdown("This model is developed by using Keras LSTM and Streamlit.")

message = st.text_area("Enter message to analyze:", placeholder="e.g., I am not happy with the delay...")

if st.button("Run Analysis"):
    if message.strip():
        # Preprocessing (Must match training max_len)
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=80)
        
        # Predict
        prediction = model.predict(padded)
        result_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = le.inverse_transform([result_idx])[0]
        
        # Display results
        st.divider()
        st.subheader(f"Predicted Sentiment: {label}")
        
        if label == 'Positive':
            st.success(f"The model is {confidence:.2f}% confident this is Positive. 😊")
        elif label == 'Negative':
            st.error(f"The model is {confidence:.2f}% confident this is Negative. 😟")
        else:
            st.info(f"The model is {confidence:.2f}% confident this is Neutral. 😐")
            
        # Show probability bar chart
        st.write("Confidence Breakdown:")
        chart_data = {
            "Sentiment": list(le.classes_),
            "Confidence": prediction[0]
        }
        st.bar_chart(data=chart_data, x="Sentiment", y="Confidence")
    else:
        st.warning("Please enter a message.")