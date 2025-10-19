# src/predict.py
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import joblib
import sys
from src.preprocess import clean_text

# Load vectorizer and model
vect = joblib.load('outputs/vectorizer.joblib')
model = joblib.load('outputs/best_model.joblib')

# Map numeric labels to readable categories
labels = {
    0: "Credit Reporting",
    1: "Debt Collection",
    2: "Consumer Loan",
    3: "Mortgage"
}

def predict(text):
    # Preprocess text
    text_clean = clean_text(text)
    x = vect.transform([text_clean])
    p = model.predict(x)
    return labels[int(p[0])]  # return readable label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"complaint text here\"")
        sys.exit(1)
    input_text = " ".join(sys.argv[1:])
    predicted_label = predict(input_text)
    print("Predicted label:", predicted_label)
