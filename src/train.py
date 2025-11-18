import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

def train_model():
    train_df = pd.read_csv("data/train.csv")
    X = train_df["text"].values
    y = train_df["label"].values

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    dump(pipeline, "model.joblib")
    print("Model trained and saved to model.joblib")

if __name__ == "__main__":
    train_model()