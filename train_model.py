import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# 1. Load Data
df = pd.read_csv('cleaned_sentiment_data.csv')

# 2. Tokenization
max_words = 5000
max_len = 80 # Increased length for better context
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['Text'])
X = pad_sequences(tokenizer.texts_to_sequences(df['Text']), maxlen=max_len)

# 3. Label Encoding
le = LabelEncoder()
y = le.fit_transform(df['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Calculate Class Weights (Fixes the Negative-to-Neutral error)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
dict_weights = dict(enumerate(weights))

# 5. Build Bidirectional LSTM
model = Sequential([
    Embedding(max_words, 128),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train with Weights
model.fit(X_train, y_train, epochs=15, batch_size=32, class_weight=dict_weights, validation_split=0.1)

# 7. Save
model.save('sentiment_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Step 2 Complete: Robust model saved.")
print(model.summary())