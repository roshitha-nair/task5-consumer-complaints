# src/preprocess.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # remove urls
    text = re.sub(r'http\S+', ' ', text)
    # keep letters and numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_product_to_label(product):
    if pd.isna(product):
        return None
    p = product.lower()
    # keywords mapping to 4 categories
    if any(k in p for k in ["credit reporting", "credit repair", "credit monitoring", "credit score"]):
        return 0
    if any(k in p for k in ["debt collection", "debt collector", "collection"]):
        return 1
    if any(k in p for k in ["consumer loan", "payday loan", "personal loan", "student loan", "auto loan"]):
        return 2
    if any(k in p for k in ["mortgage", "home loan"]):
        return 3
    return None

def load_and_prepare(csv_path, text_col='Consumer complaint narrative', label_col='Product', sample_n=None):
    """
    Load dataset, clean text, map products to labels, and split into train/test.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    print("Dataset rows:", len(df))

    # map labels
    df['label'] = df[label_col].apply(map_product_to_label)
    df['text'] = df[text_col].fillna('').apply(clean_text)

    # keep only rows with valid label and non-empty text
    df = df[df['label'].notna() & df['text'].str.len().gt(5)]

    # sample if needed
    if sample_n:
        df = df.sample(n=sample_n, random_state=42)

    X = df['text'].values
    y = df['label'].astype(int).values

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def build_vectorizer(X_train, max_features=30000):
    """
    Create and fit a TF-IDF vectorizer
    """
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vect.fit(X_train)
    return vect
