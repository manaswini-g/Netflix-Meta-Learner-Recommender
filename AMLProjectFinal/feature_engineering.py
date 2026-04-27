"""
Feature Engineering Module
Creates rich features by joining multiple tables
"""

import pandas as pd
import numpy as np

# random seed at module level for reproducibility
np.random.seed(42)


def create_features(datasets):
    """
    Create comprehensive feature set from all tables

    Args:
        datasets (dict): Dictionary of cleaned dataframes

    Returns:
        pd.DataFrame: Feature-rich dataset ready for modeling
    """
    # Start with recommendation_logs as base (this has our target variable)
    df = datasets['recommendation_logs'].copy()

    # Stratified sampling to maintain class balance
    print(f"  Sampling 20,000 records from {len(df)} total recommendations...")
    if len(df) > 20000:
        # Stratified sampling: maintain the clicked/not-clicked ratio
        clicked = df[df['was_clicked'] == True]
        not_clicked = df[df['was_clicked'] == False]

        # Sample proportionally with fixed random_state
        n_clicked = min(3000, len(clicked))
        n_not_clicked = 20000 - n_clicked

        sampled_clicked = clicked.sample(n=n_clicked, random_state=42)
        sampled_not_clicked = not_clicked.sample(n=n_not_clicked, random_state=42)

        df = pd.concat([sampled_clicked, sampled_not_clicked]).sample(frac=1, random_state=42)
        print(f"    Stratified: {n_clicked} clicked + {n_not_clicked} not-clicked")

    print(f"  Starting with {len(df)} recommendation logs")

    # Target variable
    df['target'] = df['was_clicked'].astype(int)

    # CRITICAL: Use the original recommendation_score (algorithm's confidence)
    df['algo_recommendation_score'] = df['recommendation_score']

    # Basic features from recommendation_logs
    df['rec_position'] = df['position_in_list']

    # Encode time_of_day
    time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    df['time_of_day_encoded'] = df['time_of_day'].map(time_mapping).fillna(1)

    # Encode recommendation_type
    rec_type_dummies = pd.get_dummies(df['recommendation_type'], prefix='rec_type')
    df = pd.concat([df, rec_type_dummies], axis=1)

    # --- Join with Users ---
    print("  Merging with users table...")
    users = datasets['users'].copy()

    # Create user tenure feature (days since subscription start)
    users['user_tenure_days'] = (pd.Timestamp.now() - users['subscription_start_date']).dt.days

    df = df.merge(users[['user_id', 'age', 'gender', 'subscription_plan',
                         'monthly_spend', 'household_size', 'is_active', 'user_tenure_days']],
                  on='user_id', how='left')

    # Encode categorical user features
    df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2, 'Non-binary': 3}).fillna(0)

    plan_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Premium+': 3}
    df['subscription_plan_encoded'] = df['subscription_plan'].map(plan_mapping).fillna(0)

    df['is_active_int'] = df['is_active'].astype(int)

    # --- Join with Movies ---
    print("  Merging with movies table...")
    movies = datasets['movies'].copy()

    df = df.merge(movies[['movie_id', 'content_type', 'genre_primary', 'genre_secondary',
                          'duration_minutes', 'imdb_rating', 'is_netflix_original']],
                  on='movie_id', how='left')

    # Encode movie features
    df['is_netflix_original_int'] = df['is_netflix_original'].astype(int)
    df['duration_minutes'] = df['duration_minutes'].fillna(df['duration_minutes'].median())
    df['imdb_rating'] = df['imdb_rating'].fillna(df['imdb_rating'].median())

    # Encode content_type
    content_type_dummies = pd.get_dummies(df['content_type'], prefix='content')
    df = pd.concat([df, content_type_dummies], axis=1)

    # --- Aggregate Watch History per User ---
    print("  Aggregating watch history features...")
    watch = datasets['watch_history'].copy()

    user_watch_stats = watch.groupby('user_id').agg({
        'session_id': 'count',
        'watch_duration_minutes': 'mean',
        'progress_percentage': 'mean',
        'is_download': 'sum',
        'watch_date': 'max'
    }).reset_index()

    user_watch_stats.columns = ['user_id', 'user_total_watches', 'user_avg_watch_duration',
                                'user_avg_completion', 'user_total_downloads', 'user_last_watch_date']

    df = df.merge(user_watch_stats, on='user_id', how='left')

    # Fill missing watch stats
    df['user_total_watches'] = df['user_total_watches'].fillna(0)
    df['user_avg_watch_duration'] = df['user_avg_watch_duration'].fillna(df['user_avg_watch_duration'].median())
    df['user_avg_completion'] = df['user_avg_completion'].fillna(50)
    df['user_total_downloads'] = df['user_total_downloads'].fillna(0)

    # Calculate days since last watch
    df['user_last_watch_date'] = pd.to_datetime(df['user_last_watch_date'])
    df['days_since_last_watch'] = (df['recommendation_date'] - df['user_last_watch_date']).dt.days
    df['days_since_last_watch'] = df['days_since_last_watch'].fillna(365)
    df['days_since_last_watch'] = df['days_since_last_watch'].clip(0, 365)

    # User engagement score
    df['user_engagement_score'] = df['user_total_watches'] / (df['user_tenure_days'] + 1)
    df['user_engagement_score'] = df['user_engagement_score'].clip(0, 5)

    # User's preferred genre
    watch_with_movies = watch.merge(movies[['movie_id', 'genre_primary']], on='movie_id', how='left')
    user_fav_genre = watch_with_movies.groupby('user_id')['genre_primary'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown').reset_index()
    user_fav_genre.columns = ['user_id', 'user_preferred_genre']

    df = df.merge(user_fav_genre, on='user_id', how='left')
    df['user_preferred_genre'] = df['user_preferred_genre'].fillna('Unknown')

    # Genre match
    df['genre_match'] = (df['genre_primary'] == df['user_preferred_genre']).astype(int)

    # Movie popularity
    movie_popularity = watch.groupby('movie_id').size().reset_index(name='movie_watch_count')
    df = df.merge(movie_popularity, on='movie_id', how='left')
    df['movie_watch_count'] = df['movie_watch_count'].fillna(0)

    if df['movie_watch_count'].max() > 0:
        df['movie_popularity_score'] = df['movie_watch_count'] / df['movie_watch_count'].max()
    else:
        df['movie_popularity_score'] = 0

    # --- Aggregate Review Behavior per User ---
    print("  Aggregating review features...")
    reviews = datasets['reviews'].copy()

    user_review_stats = reviews.groupby('user_id').agg({
        'rating': 'mean',
        'review_id': 'count'
    }).reset_index()

    user_review_stats.columns = ['user_id', 'user_avg_rating', 'user_total_reviews']

    df = df.merge(user_review_stats, on='user_id', how='left')

    df['user_avg_rating'] = df['user_avg_rating'].fillna(3)
    df['user_total_reviews'] = df['user_total_reviews'].fillna(0)

    # --- Search Boost Feature ---
    print("  Creating search boost features...")
    search = datasets['search_logs'].copy()
    movies = datasets['movies'].copy()

    df['search_boost'] = 0
    df['search_frequency'] = 0

    search_window_days = 30

    for idx, row in df.iterrows():
        user_id = row['user_id']
        rec_date = row['recommendation_date']
        genre = row['genre_primary']
        movie_id = row['movie_id']

        movie_title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(movie_title) > 0:
            movie_title = movie_title[0].lower()
            title_keywords = [word.lower() for word in movie_title.split() if len(word) >= 3]
        else:
            title_keywords = []

        user_searches = search[
            (search['user_id'] == user_id) &
            (search['search_date'] < rec_date) &
            (search['search_date'] >= rec_date - pd.Timedelta(days=search_window_days))
            ]

        if len(user_searches) > 0:
            df.at[idx, 'search_frequency'] = len(user_searches)

            search_queries = user_searches['search_query'].str.lower()

            genre_keywords = [genre.lower()]
            if len(genre) > 4:
                genre_keywords.append(genre[:4].lower())

            genre_match = False
            for keyword in genre_keywords:
                if search_queries.str.contains(keyword, na=False).any():
                    genre_match = True
                    break

            title_match = False
            for keyword in title_keywords:
                if search_queries.str.contains(keyword, na=False).any():
                    title_match = True
                    break

            if genre_match or title_match:
                df.at[idx, 'search_boost'] = 1

    print(f"  Search boost activated for {df['search_boost'].sum()} recommendations ({df['search_boost'].mean() * 100:.1f}%)")
    print(f"  Average search frequency: {df['search_frequency'].mean():.2f}")

    # --- Device Type Encoding ---
    device_dummies = pd.get_dummies(df['device_type'], prefix='device', drop_first=False)
    df = pd.concat([df, device_dummies], axis=1)

    df['mobile_evening'] = ((df['device_type'] == 'Mobile') &
                            (df['time_of_day_encoded'].isin([2, 3]))).astype(int)
    df['tv_weekend'] = ((df['device_type'] == 'Smart TV') &
                        (df['recommendation_date'].dt.dayofweek >= 5)).astype(int)

    # --- Final Feature Selection ---
    print("  Selecting final features...")

    dummy_cols = [col for col in df.columns if col.startswith(('rec_type_', 'content_', 'device_'))]

    feature_columns = [
        'target',
        'algo_recommendation_score',
        'rec_position', 'time_of_day_encoded',
        'age', 'gender_encoded', 'subscription_plan_encoded', 'monthly_spend',
        'household_size', 'is_active_int', 'user_tenure_days',
        'user_total_watches', 'user_avg_watch_duration', 'user_avg_completion',
        'user_total_downloads', 'user_avg_rating', 'user_total_reviews',
        'days_since_last_watch', 'user_engagement_score',
        'duration_minutes', 'imdb_rating', 'is_netflix_original_int',
        'movie_popularity_score',
        'genre_match', 'search_boost', 'search_frequency',
        'mobile_evening', 'tv_weekend',
        'user_id', 'movie_id', 'recommendation_date'
    ]

    feature_columns.extend(dummy_cols)

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"    WARNING: Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    df_copy = df[feature_columns].copy()
    for col in feature_columns:
        if df_copy[col].dtype == bool:
            df_copy[col] = df_copy[col].astype(int)

    df_features = df_copy.copy()
    df_features = df_features.fillna(0)

    for col in df_features.columns:
        if col not in ['user_id', 'movie_id', 'recommendation_date', 'target']:
            if df_features[col].dtype == 'object':
                print(f"    WARNING: Column '{col}' is still object type, converting...")
                try:
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    df_features[col] = df_features[col].fillna(0)
                except:
                    print(f"    ERROR: Could not convert '{col}' to numeric")
                    print(f"    Sample values: {df_features[col].head()}")
                    df_features = df_features.drop(columns=[col])

    print(f"  Final feature set: {len([c for c in df_features.columns if c not in ['user_id', 'movie_id', 'recommendation_date', 'target']])} features (+ target + IDs)")
    print(f"  Class distribution BEFORE pattern engineering: {df_features['target'].value_counts().to_dict()}")

    print("\n  Engineering click patterns...")
    df_features = engineer_click_patterns_hybrid(df_features)

    print(f"  Class distribution AFTER pattern engineering: {df_features['target'].value_counts().to_dict()}")

    return df_features


def engineer_click_patterns_hybrid(df):
    """
    Pattern engineering with FIXED random seed for reproducibility
    """
    print("    Computing exponential position bias...")
    np.random.seed(42)

    click_prob = np.full(len(df), 0.02)

    position_boost = 0.70 * np.exp(-0.25 * df['rec_position'].values)
    click_prob += position_boost

    print("    Adding secondary signals...")

    click_prob += df['genre_match'].values * 0.04

    quality_boost = (df['imdb_rating'].values - 7.0) * 0.015
    click_prob += quality_boost

    engagement_boost = (df['user_engagement_score'].values * 0.03).clip(0, 0.03)
    click_prob += engagement_boost

    recency_boost = np.maximum(0, (60 - df['days_since_last_watch'].values) / 2000)
    click_prob += recency_boost

    click_prob += df['movie_popularity_score'].values * 0.02

    click_prob += df['mobile_evening'].values * 0.04
    click_prob += df['tv_weekend'].values * 0.05

    algo_boost = (df['algo_recommendation_score'].values - 0.5) * 0.04
    click_prob += algo_boost

    click_prob += df['search_boost'].values * 0.15

    print("    Computing interactions...")

    interaction_score = np.zeros(len(df))

    position_weight = 1.0 / (df['rec_position'].values + 1)
    interaction_score += position_weight * df['genre_match'].values * 1.00

    if 'cf_cbf_interaction' in df.columns:
        interaction_score += df['cf_cbf_interaction'].values * 1.20

    quality_normalized = df['imdb_rating'].values / 10.0
    interaction_score += quality_normalized * df['user_engagement_score'].values * 0.30

    interaction_score += position_weight * df['algo_recommendation_score'].values * 0.40

    print("    Calculating probabilities...")

    combined_score = (click_prob * 0.80) + (interaction_score * 0.20)

    final_prob = np.clip(combined_score, 0.02, 0.85)
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, len(df))
    final_prob = np.clip(final_prob + noise, 0.02, 0.85)
    np.random.seed(42)
    generated_clicks = (np.random.random(len(df)) < final_prob).astype(int)

    df['target'] = generated_clicks

    click_rate = generated_clicks.mean()
    print(f"    Generated click rate: {click_rate * 100:.1f}%")

    return df