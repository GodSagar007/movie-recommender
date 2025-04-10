import pandas as pd

# Load movies.dat (note the engine="python" for '::' separator)
movies_df = pd.read_csv("movies.dat", 
                        sep="::", 
                        engine="python", 
                        names=["movieId", "title", "genres"])

# Save as CSV
movies_df.to_csv("movies.csv", index=False)

print("✅ movies.csv saved to data/")
