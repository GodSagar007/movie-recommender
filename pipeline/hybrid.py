
import numpy as np
from pipeline.utils import compute_genre_similarity_matrix

def hybrid_score(user_id, movie_id, model, interactions_df, genre_similarity, movies_df):
    svd_score = model.predict(user_id, movie_id).est

    watched = interactions_df[interactions_df['userId'] == user_id]['movieId'].values
    if len(watched) == 0:
        return svd_score

    idx_map = {mid: i for i, mid in enumerate(movies_df.index)}
    movie_idx = idx_map.get(movie_id)
    if movie_idx is None:
        return svd_score

    similarities = []
    for mid in watched:
        if mid in idx_map:
            similarities.append(genre_similarity[movie_idx][idx_map[mid]])
    genre_score = np.mean(similarities) if similarities else 0

    return 0.7 * svd_score + 0.3 * genre_score
