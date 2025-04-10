import joblib
from surprise import accuracy

MODEL_PATH = "models/recommender.pkl"
TESTSET_PATH = "models/testset.pkl"

def load_model_and_testset():
    model = joblib.load(MODEL_PATH)
    testset = joblib.load(TESTSET_PATH)
    return model, testset

def evaluate_on_testset(model, testset):
    print("Evaluating model on test set...")
    predictions = model.test(testset)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    print("\nSample Predictions:")
    for pred in predictions[:5]:
        print(f"User {pred.uid} → Item {pred.iid} | Actual: {pred.r_ui}, Predicted: {pred.est:.2f}")

if __name__ == "__main__":
    model, testset = load_model_and_testset()
    evaluate_on_testset(model, testset)
