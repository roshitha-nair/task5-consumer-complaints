# notebooks/eda_script.py
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Add the src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocess import map_product_to_label, clean_text

# Paths
DATA_PATH = r"C:\roshitha\task5-consumer-complaints\data\consumer_complaints.csv"
OUTPUT_DIR = r"C:\roshitha\task5-consumer-complaints\outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Map product categories
df['mapped'] = df['Product'].apply(map_product_to_label)
print("Original rows:", len(df))
print("Mapped value counts:")
print(df['mapped'].value_counts(dropna=False))

# Filter and clean text
df['text'] = df['Consumer complaint narrative'].fillna('').apply(clean_text)
df = df[df['mapped'].notna() & df['text'].str.len().gt(5)]
print("Rows after cleaning:", len(df))

# Class distribution plot
counts = df['mapped'].value_counts().sort_index()
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values)
plt.title("Class distribution (mapped labels)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
plt.text(0.95, 0.01, f"Roshitha B Nair | {now}", 
         fontsize=8, color='gray', 
         ha='right', va='bottom', transform=plt.gca().transAxes)
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print(f"Saved {os.path.join(OUTPUT_DIR, 'class_distribution.png')}")

# Top words per class using CountVectorizer
for label in sorted(df['mapped'].unique()):
    texts = df[df['mapped']==label]['text'].sample(
        n=min(5000, df[df['mapped']==label].shape[0]), 
        random_state=42
    )
    cv = CountVectorizer(max_features=1000, ngram_range=(1,1), stop_words='english')
    X = cv.fit_transform(texts)
    sums = X.sum(axis=0)
    words = [(w, sums[0, idx]) for w, idx in cv.vocabulary_.items()]
    words_sorted = sorted(words, key=lambda x: x[1], reverse=True)[:20]
    top_words = [w for w,c in words_sorted]

    # Plot horizontal bar chart
    plt.figure(figsize=(8,6))
    plt.barh(top_words[::-1], [c for w,c in words_sorted][::-1])
    plt.title(f"Top tokens (label={label})")
    plt.tight_layout()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(0.95, 0.01, f"Roshitha B Nair | {now}", 
         fontsize=8, color='gray', 
         ha='right', va='bottom', transform=plt.gca().transAxes)
    file_path = os.path.join(OUTPUT_DIR, f"top_tokens_label_{label}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"Saved {file_path}")

# Save sample texts per class for README
for label in sorted(df['mapped'].unique()):
    sample_texts = df[df['mapped']==label]['text'].head(5).tolist()
    with open(os.path.join(OUTPUT_DIR, f"samples_label_{label}.txt"), "w", encoding="utf8") as f:
        for t in sample_texts:
            f.write(t + "\n\n")
print("Saved sample texts in outputs folder.")

