import pickle
import numpy as np
from tensorflow.keras.models import load_model

def verify_assets():
    print("--- Starting File Verification ---")
    try:
        # 1. Check Model
        model = load_model('sentiment_model.h5')
        print("✅ sentiment_model.h5: Loaded Successfully")
        print(f"   Model Input Shape: {model.input_shape}")
        
        # 2. Check Tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ tokenizer.pkl: Loaded Successfully")
        print(f"   Vocabulary Size: {len(tokenizer.word_index)} words")
        
        # 3. Check Label Encoder
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        print("✅ label_encoder.pkl: Loaded Successfully")
        print(f"   Categories Detected: {list(le.classes_)}")
        
        print("\n--- All files are ready for app.py! ---")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        print("Hint: Make sure you ran preprocess.py and train_model.py first.")

if __name__ == "__main__":
    verify_assets()