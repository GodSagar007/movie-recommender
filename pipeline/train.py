import joblib
from .preprocess import load_interactions, prepare_surprise_dataset
from surprise import SVD
from surprise.model_selection import train_test_split
import os

MODEL_PATH = "models/recommender.pkl"
TESTSET_PATH = "models/testset.pkl"

def run():
    df = load_interactions()
    data = prepare_surprise_dataset(df)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("Training model...")
    model = SVD()
    model.fit(trainset)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(testset, TESTSET_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test set saved to {TESTSET_PATH}")
