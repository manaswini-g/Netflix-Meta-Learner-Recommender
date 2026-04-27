"""
Content-Based Filtering Module
Implements TF-IDF vectorization and cosine similarity
Compares different feature combinations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set random seed at module level
np.random.seed(42)


def create_movie_profiles(movies_df, approach='multi_feature'):
    """
    Create movie content profiles using different approaches
    """
    if approach == 'genre_only':
        movies_df['content_text'] = movies_df['genre_primary'] + ' ' + movies_df['genre_secondary']

    elif approach == 'multi_feature':
        movies_df['content_text'] = (
                movies_df['genre_primary'] + ' ' +
                movies_df['genre_secondary'] + ' ' +
                movies_df['content_type'] + ' ' +
                movies_df['language']
        )

    elif approach == 'hybrid':
        movies_df['content_text'] = (
                movies_df['genre_primary'] + ' ' +
                movies_df['genre_secondary'] + ' ' +
                movies_df['content_type'] + ' ' +
                movies_df['language']
        )

    return movies_df


def compute_similarity_matrix(movies_df, approach='multi_feature'):
    """
    Compute movie-movie similarity matrix
    """
    print(f"    Computing similarity with approach: {approach}")

    movies_df = create_movie_profiles(movies_df.copy(), approach)

    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf.fit_transform(movies_df['content_text'])

    if approach == 'hybrid':
        numerical_features = movies_df[['imdb_rating', 'duration_minutes']].fillna(0).values
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(numerical_features)

        tfidf_dense = tfidf_matrix.toarray()
        combined_features = np.hstack([
            tfidf_dense * 0.7,
            numerical_features * 0.3
        ])

        similarity_matrix = cosine_similarity(combined_features)
    else:
        similarity_matrix = cosine_similarity(tfidf_matrix)

    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movie_id'])}

    return similarity_matrix, movie_id_to_idx, tfidf


def get_user_movie_similarity(user_id, movie_id, watch_history_df, movies_df,
                              similarity_matrix, movie_id_to_idx):
    """
    Calculate similarity between a movie and user's watch history
    """
    user_watches = watch_history_df[watch_history_df['user_id'] == user_id]['movie_id'].unique()

    if len(user_watches) == 0:
        return 0.5

    if movie_id not in movie_id_to_idx:
        return 0.5

    target_idx = movie_id_to_idx[movie_id]

    similarities = []
    for watched_movie in user_watches:
        if watched_movie in movie_id_to_idx:
            watched_idx = movie_id_to_idx[watched_movie]
            sim_score = similarity_matrix[target_idx, watched_idx]
            similarities.append(sim_score)

    if len(similarities) == 0:
        return 0.5

    return np.mean(similarities)


def compare_cbf_approaches(datasets, train_df, test_df):
    """
    Compare different CBF approaches and return the best one
    """
    movies = datasets['movies'].copy()
    watch_history = datasets['watch_history'].copy()

    approaches = ['genre_only', 'multi_feature', 'hybrid']
    results = {}

    for approach in approaches:
        print(f"\n    Testing approach: {approach}")

        sim_matrix, movie_idx_map, tfidf = compute_similarity_matrix(movies, approach)

        test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)

        scores = []
        for _, row in test_sample.iterrows():
            score = get_user_movie_similarity(
                row['user_id'],
                row['movie_id'],
                watch_history,
                movies,
                sim_matrix,
                movie_idx_map
            )
            scores.append(score)

        correlation = np.corrcoef(scores, test_sample['target'])[0, 1]

        print(f"      Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"      Correlation with clicks: {correlation:.4f}")

        results[approach] = {
            'sim_matrix': sim_matrix,
            'movie_idx_map': movie_idx_map,
            'tfidf': tfidf,
            'correlation': correlation
        }

    best_approach = max(results.keys(), key=lambda k: results[k]['correlation'])
    print(f"\n    Best approach: {best_approach} (correlation: {results[best_approach]['correlation']:.4f})")

    return results[best_approach], best_approach


def train_content_based_model(datasets, train_df, test_df):
    """
    Main function to train content-based filtering
    """
    print("\n  Comparing CBF feature combinations...")

    best_cbf_result, best_approach = compare_cbf_approaches(datasets, train_df, test_df)

    print("\n  Generating CBF scores for full train/test sets...")
    movies = datasets['movies'].copy()
    watch_history = datasets['watch_history'].copy()

    sim_matrix = best_cbf_result['sim_matrix']
    movie_idx_map = best_cbf_result['movie_idx_map']

    train_scores = []
    for _, row in train_df.iterrows():
        score = get_user_movie_similarity(
            row['user_id'],
            row['movie_id'],
            watch_history,
            movies,
            sim_matrix,
            movie_idx_map
        )
        train_scores.append(score)

    test_scores = []
    for _, row in test_df.iterrows():
        score = get_user_movie_similarity(
            row['user_id'],
            row['movie_id'],
            watch_history,
            movies,
            sim_matrix,
            movie_idx_map
        )
        test_scores.append(score)

    os.makedirs('models', exist_ok=True)
    with open('models/best_cbf_model.pkl', 'wb') as f:
        pickle.dump({
            'sim_matrix': sim_matrix,
            'movie_idx_map': movie_idx_map,
            'tfidf': best_cbf_result['tfidf'],
            'approach': best_approach
        }, f)

    return {
        'best_approach': best_approach,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'sim_matrix': sim_matrix
    }