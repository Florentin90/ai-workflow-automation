import pandas as pd
from sklearn.model_selection import train_test_split

def generate_dataset():
    # Synthetic dataset for demo
    data = {
        "text": [
            "hello world",
            "machine learning is fun",
            "aws lambda scales",
            "sign language accessibility matters",
            "python scripts automate workflows",
            "generative ai creates content",
            "docker containers are portable",
            "ci cd pipelines deploy fast",
            "okta and azure ad manage identity",
            "monitoring with azure keeps systems healthy"
        ],
        "label": [0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  # 1 = AI/Tech, 0 = Infra/Ops
    }

    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Generated dataset: data/train.csv and data/test.csv")

if __name__ == "__main__":
    generate_dataset()