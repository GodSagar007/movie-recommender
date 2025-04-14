
import pandas as pd
from surprise import SVD, Dataset, Reader, dump

def train_model(interactions_csv):
    df = pd.read_csv(interactions_csv)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    dump.dump('models/svd_model.pkl', algo=model)

if __name__ == "__main__":
    train_model("data/interactions.csv")
