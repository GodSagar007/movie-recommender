
import pandas as pd

def load_movie_genres(movies_path):
    df = pd.read_csv(movies_path, sep='::', engine='python', names=['movieId', 'title', 'genres'])
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))
    return df.set_index('movieId')
