from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from surprise import SVD

# Load the trained recommendation model
model = joblib.load("models/recommender.pkl")

# Load the movie metadata and build a mapping of movieId -> title
movies_df = pd.read_csv("data/movies.csv")
movie_id_to_title = {
    str(movieId): title for movieId, title in zip(movies_df['movieId'], movies_df['title'])
}

# Initialize FastAPI app
app = FastAPI()

# Define the request schema
class UserInput(BaseModel):
    user_id: int
    top_n: int = 5

@app.post("/recommend")
def recommend(input_data: UserInput):
    user_id = str(input_data.user_id)
    top_n = input_data.top_n

    # Get all movie IDs the model knows about
    all_items = model.trainset.all_items()
    all_item_ids = [model.trainset.to_raw_iid(iid) for iid in all_items]

    # Predict ratings for each movie
    predictions = [(iid, model.predict(user_id, iid).est) for iid in all_item_ids]
    top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    # Format results with movie titles
    recommendations = [
        {
            "movie_id": movie_id,
            "title": movie_id_to_title.get(str(movie_id), "Unknown"),
            "predicted_rating": round(score, 2)
        }
        for movie_id, score in top_preds
    ]

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }
