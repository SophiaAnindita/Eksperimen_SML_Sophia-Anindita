import pandas as pd
import re
import string

def load_data(path):
    return pd.read_csv(path, encoding='latin-1')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def preprocess(df):
    # ambil kolom penting
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # hapus duplikasi
    df = df.drop_duplicates()

    # cleaning text
    df['clean_text'] = df['text'].apply(clean_text)

    # encoding label
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data('spam.csv')
    df = preprocess(df)
    save_data(df, 'sms_clean.csv')

    print("Preprocessing selesai!")
