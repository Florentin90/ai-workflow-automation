from joblib import load

EXAMPLES = [
    "generative ai helps accessibility",
    "azure ad and okta secure identities",
    "ci cd pipelines accelerate deployments",
    "docker makes environments reproducible"
]

def run_inference():
    model = load("model.joblib")
    preds = model.predict(EXAMPLES)
    for text, pred in zip(EXAMPLES, preds):
        label = "AI/Tech" if pred == 1 else "Infra/Ops"
        print(f"{text} -> {label}")

if __name__ == "__main__":
    run_inference()