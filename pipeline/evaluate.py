from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from .preprocess import load_interactions, prepare_surprise_dataset

def evaluate():
    df = load_interactions()
    data = prepare_surprise_dataset(df)

    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)

    print("Evaluation Metrics:")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

if __name__ == "__main__":
    evaluate()
