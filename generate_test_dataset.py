# synthetic_generate.py
import pandas as pd
import openai
from typing import List, Dict
import json
import time
import re
from collections import Counter

OPENAI_API_KEY = #YOUR_API_KEY
OUTPUT_FILE = 'test_queries.json'
LLM_MODEL = "gpt-4o-mini"
CSV_PATH = 'data/TMDB_all_movies.csv'
N_SEED_MOVIES = 80
INTERMEDIATE_FILE = 'test_queries_partial.json'
SAVE_EVERY_N_QUERIES = 20
REQUESTS_PER_BATCH = 3
SLEEP_AFTER_BATCH = 20
RETRY_SLEEP = 20

openai.api_key = OPENAI_API_KEY


class SyntheticQueryGenerator:
    def __init__(self, llm_model=LLM_MODEL):
        self.llm_model = llm_model

    def generate_queries_for_movie(self, movie: Dict, attempt: int = 1) -> List[Dict]:
        prompt = fprompt = f"""
You watched a movie a long time ago and now you're trying to remember its name. 
Write exactly ONE natural sentence in English — exactly how a real person would describe it to a friend or a movie bot.

Use these clues (but NEVER mention the title!):

- Year: around {movie['release_date'][:4]}
- Genres: {', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie.get('genres', 'unknown')}
- Plot: {movie['overview']}

Make it sound casual and human. Include 2–3 specific details so the movie is clearly identifiable, but still like you're struggling to remember.

Good examples:
- "that 90s action movie where the guy has to stop a bomb on a bus that can't slow down"
- "early 2000s animated film about a fish who gets lost and his dad swims across the ocean"
- "2010s sci-fi with leonardo dicaprio where they go into people's dreams to steal ideas"
- "late 70s space movie with a farm boy, a princess and a walking carpet"

Now write just ONE sentence for the movie above.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.95,
                max_tokens=120,
                response_format={"type": "text"}
            )
            raw = response.choices[0].message.content.strip().strip('"\'')
            

            query_item = {
                "query": raw,
                "type": "memory_recall",
                "difficulty": "medium",
                "explanation": "natural user recall by plot"
            }
            self._add_gold_truth([query_item], movie)
            return [query_item]

        except openai.error.RateLimitError as e:
            print(f"  Rate limit hit (attempt {attempt}). Waiting {RETRY_SLEEP}s...")
            time.sleep(RETRY_SLEEP)
            return self.generate_queries_for_movie(movie, attempt + 1)

        except Exception as e:
            print(f"  Unexpected error for {movie['title']}: {e}")
            return []

    def _add_gold_truth(self, queries: List[Dict], movie: Dict):
        for q in queries:
            q['gold_movies'] = [{
                'tmdb_id': int(movie['id']),
                'title': movie['title'],
                'relevance': 5
            }]


def get_movie_data_from_tmdb(df: pd.DataFrame, movie_id: int) -> Dict:
    movie_row = df[df['id'] == movie_id].iloc[0]
    genres = movie_row['genres'].split(', ') if pd.notna(movie_row['genres']) and movie_row['genres'] else []
    return {
        'id': movie_row['id'],
        'title': movie_row['title'],
        'overview': movie_row.get('overview', ''),
        'genres': genres,
        'release_date': movie_row.get('release_date', ''),
        'vote_average': movie_row.get('vote_average', 0),
    }


def select_diverse_seed_movies(df: pd.DataFrame, n_movies=N_SEED_MOVIES) -> List[int]:
    df = df.dropna(subset=['vote_count', 'vote_average', 'overview'])
    df = df[df['overview'].str.len() > 50]
    df['score'] = df['vote_count'] * df['vote_average']
    top_ids = df.nlargest(n_movies, 'score')['id'].tolist()
    return top_ids


def save_intermediate(queries: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


def generate_test_dataset(csv_path=CSV_PATH, n_seed_movies=N_SEED_MOVIES, output_file=OUTPUT_FILE):

    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df[df['overview'].str.len() > 20]

    seed_ids = select_diverse_seed_movies(df, n_seed_movies)
    
    seed_movies = []
    for mid in seed_ids:
        try:
            movie = get_movie_data_from_tmdb(df, mid)
            seed_movies.append(movie)
            print(f"  {movie['title']} ({movie['release_date'][:4]})")
        except:
            pass
    
    generator = SyntheticQueryGenerator()
    all_queries = []
    idx = 0
    
    for i, movie in enumerate(seed_movies, 1):
        print(f"  [{i}/{len(seed_movies)}] {movie['title']}")
        queries = generator.generate_queries_for_movie(movie)
        if queries:
            all_queries.extend(queries)
            print(f"query\"{queries[0]['query']}\"")
        
        idx += 1
        if idx % REQUESTS_PER_BATCH == 0:
            print(f"Pause for {SLEEP_AFTER_BATCH} sec...")
            time.sleep(SLEEP_AFTER_BATCH)

        if len(all_queries) % SAVE_EVERY_N_QUERIES == 0 and all_queries:
            save_intermediate(all_queries, INTERMEDIATE_FILE)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    
    return all_queries


if __name__ == "__main__":
    dataset = generate_test_dataset()