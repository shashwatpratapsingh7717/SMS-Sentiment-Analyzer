import pandas as pd

def simplify_sentiment(label):
    label = str(label).strip().lower()
    
    # Define keywords for each category
    pos_keywords = ['positive', 'joy', 'happy', 'excited', 'love', 'grateful', 'success', 'awe', 'inspiration', 'pride', 'content']
    neg_keywords = ['negative', 'anger', 'fear', 'sad', 'disgust', 'disappoint', 'hate', 'bitter', 'frustrat', 'lonely', 'anxiety', 'bad', 'grief']
    
    if any(word in label for word in neg_keywords):
        return 'Negative'
    elif any(word in label for word in pos_keywords):
        return 'Positive'
    else:
        return 'Neutral'

# Load and process
df = pd.read_csv('sentimentdataset.csv')
df['Text'] = df['Text'].str.strip()
df['Target'] = df['Sentiment'].apply(simplify_sentiment)

# Check the distribution to see if it's more balanced now
print("New Distribution:\n", df['Target'].value_counts())

df.to_csv('cleaned_sentiment_data.csv', index=False)
print("Step 1 Complete: Cleaned data saved.")