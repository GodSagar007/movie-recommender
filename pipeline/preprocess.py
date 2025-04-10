import pandas as pd
from surprise import Dataset, Reader

def load_interactions(path="data/u.data"):
    df = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    print(f"Loaded {len(df)} interaction records.")
    return df

def prepare_surprise_dataset(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
    return data
