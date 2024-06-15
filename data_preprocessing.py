import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    data = load_data('data/hate_speech.csv')
    data = preprocess_data(data)
    X = data['cleaned_text']
    y = data['label']
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Save the preprocessed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)
