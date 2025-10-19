# src/train.py
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import load_and_prepare, build_vectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

def train_pipeline(csv_path, sample_n=None):
    X_train, X_test, y_train, y_test = load_and_prepare(csv_path, sample_n=sample_n)
    vect = build_vectorizer(X_train, max_features=30000)
    Xtr = vect.transform(X_train)
    Xte = vect.transform(X_test)

    models = {
        'logreg': LogisticRegression(max_iter=500, class_weight='balanced'),
        'rf': RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced'),
        'xgb': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    }
    results = {}
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    for name, m in models.items():
        print("Training", name)
        m.fit(Xtr, y_train)
        preds = m.predict(Xte)
        print(name, "classification report:")
        print(classification_report(y_test, preds))
        results[name] = (m, classification_report(y_test, preds, output_dict=True))
        # save confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Pred")
        plt.ylabel("True")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.95, 0.01, f"Roshitha B Nair | {now}", 
         fontsize=8, color='gray', 
         ha='right', va='bottom', transform=plt.gca().transAxes)
        plt.savefig(f"outputs/confusion_{name}.png", dpi=150)
        plt.close()
    
    # Save all models
    for name, (m, _) in results.items():
        joblib.dump(m, f"outputs/{name}_model.joblib")

    # Choose best model by average F1 (simple heuristic)
    best = None
    best_score = -1
    for name, (m, report) in results.items():
        # macro avg f1-score:
        f1 = report.get('macro avg', {}).get('f1-score', 0)
        if f1 > best_score:
            best_score = f1
            best = name

    print("Best model:", best, "score:", best_score)

    # save best model and vectorizer
    joblib.dump(results[best][0], "outputs/best_model.joblib")
    joblib.dump(vect, "outputs/vectorizer.joblib")
    print("Saved model and vectorizer to outputs/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/train.py data/consumer_complaints.csv [sample_n]")
        sys.exit(1)
    csv_path = sys.argv[1]
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    train_pipeline(csv_path, sample_n)
