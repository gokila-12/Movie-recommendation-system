import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
print("🔹 Loading TMDB data...")
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# --- Merge data on title ---
movies = movies.merge(credits, on='title')

# --- Select useful columns ---
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# --- Helper functions ---

def convert(obj):
    """Convert stringified list of dicts into a list of names."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def get_director(obj):
    """Extract director name from crew."""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def collapse(L):
    """Remove spaces inside multi-word names."""
    return [i.replace(" ", "") for i in L]

# --- Apply data cleaning ---
print("🔹 Cleaning and processing columns...")

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])   # top 3 actors
movies['crew'] = movies['crew'].apply(get_director)

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# --- Combine all text features ---
movies['tags'] = movies.apply(
    lambda row: row['overview'].split() + row['genres'] + row['keywords'] + row['cast'] + row['crew'],
    axis=1
)

# --- Build new dataframe ---
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# --- Vectorization ---
print("🔹 Vectorizing text data (this may take a few seconds)...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# --- Compute similarity ---
print("🔹 Calculating cosine similarity matrix...")
similarity = cosine_similarity(vectors)

# --- Save model ---
print("🔹 Saving model and data...")
movie_dict = new_df.to_dict()
pickle.dump(movie_dict, open('data/movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('data/similarity.pkl', 'wb'))

print("✅ Movie data and similarity model saved successfully!")
print(f"📁 Saved to: data/movie_dict.pkl and data/similarity.pkl")
print(f"🎬 Total movies processed: {len(new_df)}")
