import pickle
import pandas as pd
import requests
import time

API_KEY = "831dc5cf6ed482400a217ee7228961a1"
MOVIE_DICT_FILE = 'data/movie_dict.pkl'
OUTPUT_FILE = 'data/movie_dict_with_posters_genres.pkl'
PLACEHOLDER_POSTER = "https://placehold.co/300x450/333/FFFFFF?text=No+Poster"

movies_dict = pickle.load(open(MOVIE_DICT_FILE, 'rb'))
movies = pd.DataFrame(movies_dict)

def fetch_movie_data(movie_id):
    poster_url = PLACEHOLDER_POSTER
    genres_list = []

    if not movie_id or movie_id == 0:
        return poster_url, genres_list

    for attempt in range(3):
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            data = requests.get(url, timeout=5).json()
            poster_path = data.get("poster_path")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            genres = data.get("genres", [])
            genres_list = [g['name'] for g in genres]
            break
        except Exception as e:
            print(f"Error fetching movie ID {movie_id}, attempt {attempt+1}: {e}")
            time.sleep(1)

    return poster_url, genres_list

posters = []
genres = []
total_movies = len(movies)

for idx, movie_id in enumerate(movies['movie_id']):
    try:
        poster, genre_list = fetch_movie_data(movie_id)
    except Exception as e:
        print(f"Skipping movie ID {movie_id} due to error: {e}")
        poster, genre_list = PLACEHOLDER_POSTER, []

    posters.append(poster)
    genres.append(genre_list)

    # Save progress every 50 movies
    if (idx+1) % 50 == 0:
        print(f"Processed {idx+1}/{total_movies} movies")
        pickle.dump({'poster': posters, 'genres': genres}, open(OUTPUT_FILE, 'wb'))

    time.sleep(0.1)

# Final save
movies['poster'] = posters
movies['genres'] = genres
pickle.dump(movies.to_dict(orient='list'), open(OUTPUT_FILE, 'wb'))
print(f"Done! Preloaded posters and genres saved to {OUTPUT_FILE}")
