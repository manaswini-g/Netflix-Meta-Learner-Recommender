"""
Collaborative Filtering Module
Implements Matrix Factorization with sklearn
Compares TruncatedSVD and KNN approaches
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix
import pickle
import os

# Set random seed at module level
np.random.seed(42)


def create_interaction_matrix(datasets):
    """
    Create user-item interaction matrix combining explicit and implicit ratings

    Returns:
        rating_df: DataFrame with user_id, movie_id, rating
        user_mapper: dict mapping user_id to matrix index
        movie_mapper: dict mapping movie_id to matrix index
    """
    print("  Creating interaction matrix...")

    reviews = datasets['reviews'].copy()
    explicit_ratings = reviews[['user_id', 'movie_id', 'rating']].copy()
    explicit_ratings['source'] = 'explicit'

    watch = datasets['watch_history'].copy()

    watch['implicit_rating'] = (watch['progress_percentage'] / 100) * 4 + 1
    watch['implicit_rating'] = watch['implicit_rating'].clip(1, 5)

    implicit_ratings = watch[['user_id', 'movie_id', 'implicit_rating']].copy()
    implicit_ratings.columns = ['user_id', 'movie_id', 'rating']
    implicit_ratings['source'] = 'implicit'

    all_ratings = pd.concat([explicit_ratings, implicit_ratings], ignore_index=True)
    all_ratings = all_ratings.sort_values('source')
    all_ratings = all_ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')

    print(
        f"    Total ratings: {len(all_ratings)} ({len(explicit_ratings)} explicit + {len(all_ratings) - len(explicit_ratings)} implicit)")

    unique_users = all_ratings['user_id'].unique()
    unique_movies = all_ratings['movie_id'].unique()

    user_mapper = {user: idx for idx, user in enumerate(unique_users)}
    movie_mapper = {movie: idx for idx, movie in enumerate(unique_movies)}

    rating_df = all_ratings[['user_id', 'movie_id', 'rating']].copy()

    return rating_df, user_mapper, movie_mapper


def create_sparse_matrix(rating_df, user_mapper, movie_mapper):
    """
    Create sparse user-item matrix
    """
    n_users = len(user_mapper)
    n_movies = len(movie_mapper)

    user_indices = rating_df['user_id'].map(user_mapper).values
    movie_indices = rating_df['movie_id'].map(movie_mapper).values
    ratings = rating_df['rating'].values

    user_item_matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(n_users, n_movies)
    )

    return user_item_matrix


def train_svd_model(user_item_matrix):
    """
    Train TruncatedSVD model with different n_components
    """
    print("    Training TruncatedSVD with different components...")

    best_rmse = float('inf')
    best_n_components = None
    best_model = None

    for n_components in [50, 100, 150]:
        print(f"      Testing n_components={n_components}...")

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(user_item_matrix)
        movie_factors = svd.components_.T

        predicted_matrix = np.dot(user_factors, movie_factors.T)

        non_zero_mask = user_item_matrix.toarray() > 0
        actual = user_item_matrix.toarray()[non_zero_mask]
        predicted = predicted_matrix[non_zero_mask]

        predicted = np.clip(predicted, 1, 5)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"        RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_components = n_components
            best_model = svd

    print(f"    Best SVD: n_components={best_n_components}, RMSE={best_rmse:.4f}")

    return best_model, best_rmse, best_n_components, user_item_matrix


def train_knn_model(user_item_matrix):
    """
    Train KNN model with different k values
    """
    print("    Training KNN with different k values...")

    best_rmse = float('inf')
    best_k = None
    best_model = None

    for k in [20, 40, 60]:
        print(f"      Testing k={k}...")

        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(user_item_matrix)

        dense_matrix = user_item_matrix.toarray()
        predictions = []
        actuals = []

        # FIXED random seed for sampling
        np.random.seed(42)
        sample_users = np.random.choice(user_item_matrix.shape[0],
                                        min(1000, user_item_matrix.shape[0]),
                                        replace=False)

        for user_idx in sample_users:
            user_vector = user_item_matrix[user_idx:user_idx + 1]

            distances, indices = knn.kneighbors(user_vector)

            user_ratings = dense_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]

            if len(rated_items) == 0:
                continue

            for item_idx in rated_items:
                neighbor_ratings = dense_matrix[indices[0], item_idx]
                neighbor_ratings = neighbor_ratings[neighbor_ratings > 0]

                if len(neighbor_ratings) > 0:
                    pred = neighbor_ratings.mean()
                    predictions.append(pred)
                    actuals.append(user_ratings[item_idx])

        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            print(f"        RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
                best_model = knn

    print(f"    Best KNN: k={best_k}, RMSE={best_rmse:.4f}")

    return best_model, best_rmse, best_k


def get_predictions_svd(svd_model, user_item_matrix, df, user_mapper, movie_mapper):
    """
    Get SVD predictions for user-movie pairs in df
    """
    print("    Generating SVD predictions...")

    user_factors = svd_model.transform(user_item_matrix)
    movie_factors = svd_model.components_.T

    predictions = []
    avg_rating = 3.0

    for _, row in df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']

        if user_id in user_mapper and movie_id in movie_mapper:
            user_idx = user_mapper[user_id]
            movie_idx = movie_mapper[movie_id]

            pred = np.dot(user_factors[user_idx], movie_factors[movie_idx])
            pred = np.clip(pred, 1, 5)
            predictions.append(pred)
        else:
            predictions.append(avg_rating)

    predictions = np.array(predictions)
    predictions = (predictions - 1) / 4

    return predictions


def get_predictions_knn(knn_model, user_item_matrix, df, user_mapper, movie_mapper):
    """
    Get KNN predictions for user-movie pairs in df
    """
    print("    Generating KNN predictions...")

    dense_matrix = user_item_matrix.toarray()
    predictions = []
    avg_rating = 3.0

    for _, row in df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']

        if user_id in user_mapper and movie_id in movie_mapper:
            user_idx = user_mapper[user_id]
            movie_idx = movie_mapper[movie_id]

            user_vector = user_item_matrix[user_idx:user_idx + 1]
            distances, indices = knn_model.kneighbors(user_vector)

            neighbor_ratings = dense_matrix[indices[0], movie_idx]
            neighbor_ratings = neighbor_ratings[neighbor_ratings > 0]

            if len(neighbor_ratings) > 0:
                pred = neighbor_ratings.mean()
                pred = np.clip(pred, 1, 5)
                predictions.append(pred)
            else:
                predictions.append(avg_rating)
        else:
            predictions.append(avg_rating)

    predictions = np.array(predictions)
    predictions = (predictions - 1) / 4

    return predictions


def train_collaborative_models(datasets, train_df, test_df):
    """
    Main function to train and compare CF models

    Returns:
        dict: Results including best model, predictions, and metrics
    """
    rating_df, user_mapper, movie_mapper = create_interaction_matrix(datasets)
    user_item_matrix = create_sparse_matrix(rating_df, user_mapper, movie_mapper)

    print(f"    Matrix shape: {user_item_matrix.shape} (users x movies)")
    print(
        f"    Matrix sparsity: {(1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%")

    print("\n  [CF Model 1: TruncatedSVD]")
    svd_model, svd_rmse, svd_params, _ = train_svd_model(user_item_matrix)

    print("\n  [CF Model 2: KNN]")
    knn_model, knn_rmse, knn_params = train_knn_model(user_item_matrix)

    if svd_rmse <= knn_rmse:
        best_model_name = 'SVD'
        best_rmse = svd_rmse
        best_params = {'n_components': svd_params}

        train_predictions = get_predictions_svd(svd_model, user_item_matrix,
                                                train_df, user_mapper, movie_mapper)
        test_predictions = get_predictions_svd(svd_model, user_item_matrix,
                                               test_df, user_mapper, movie_mapper)
    else:
        best_model_name = 'KNN'
        best_rmse = knn_rmse
        best_params = {'k': knn_params}

        train_predictions = get_predictions_knn(knn_model, user_item_matrix,
                                                train_df, user_mapper, movie_mapper)
        test_predictions = get_predictions_knn(knn_model, user_item_matrix,
                                               test_df, user_mapper, movie_mapper)

    print(f"\n  Winner: {best_model_name} (RMSE: {best_rmse:.4f})")

    os.makedirs('models', exist_ok=True)
    if best_model_name == 'SVD':
        with open('models/best_cf_model_svd.pkl', 'wb') as f:
            pickle.dump({
                'model': svd_model,
                'user_item_matrix': user_item_matrix,
                'user_mapper': user_mapper,
                'movie_mapper': movie_mapper
            }, f)
    else:
        with open('models/best_cf_model_knn.pkl', 'wb') as f:
            pickle.dump({
                'model': knn_model,
                'user_item_matrix': user_item_matrix,
                'user_mapper': user_mapper,
                'movie_mapper': movie_mapper
            }, f)

    return {
        'best_model_name': best_model_name,
        'best_rmse': best_rmse,
        'best_params': best_params,
        'svd_rmse': svd_rmse,
        'knn_rmse': knn_rmse,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }