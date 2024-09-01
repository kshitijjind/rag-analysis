import pandas as pd


def loadDataset():
    # Load dataset
    df = pd.read_json("hf://datasets/Abirate/english_quotes/quotes.jsonl", lines=True)

    return df.to_dict(orient="records")