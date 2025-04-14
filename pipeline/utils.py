
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_genre_similarity_matrix(movies_df):
    genre_strings = movies_df['genres'].apply(lambda genres: ' '.join(genres))
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(genre_strings)
    similarity = cosine_similarity(tfidf_matrix)
    return similarity
