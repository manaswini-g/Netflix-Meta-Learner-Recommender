"""
Data Loading and Cleaning Module
Loads all 6 CSV files and performs cleaning operations
"""

import pandas as pd
import os

DATASET_DIR = 'datasets'
FILENAMES = ["users.csv", "movies.csv", "watch_history.csv",
             "recommendation_logs.csv", "search_logs.csv", "reviews.csv"]


def load_and_clean_data():
    """
    Load all datasets and perform cleaning operations

    Returns:
        dict: Dictionary containing all cleaned dataframes
    """
    print("  Loading datasets from 'datasets/' folder...")
    dfs = {}

    # Load all CSV files
    for filename in FILENAMES:
        file_path = os.path.join(DATASET_DIR, filename)
        try:
            df = pd.read_csv(file_path)
            df_key = filename.replace(".csv", "")
            dfs[df_key] = df
            print(f"    ✓ Loaded: {filename}")
        except FileNotFoundError:
            print(f" Error: File not found at '{file_path}'")
            raise
        except Exception as e:
            print(f" Error loading {filename}: {e}")
            raise

    print("\n  Cleaning datasets...")

    # Clean users.csv
    df = dfs["users"].copy()
    df = df.drop_duplicates(subset=['user_id'], keep='first')
    df['subscription_start_date'] = pd.to_datetime(df['subscription_start_date'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['age'] = df['age'].clip(lower=0)
    df['age'] = df['age'].fillna(df['age'].median()).astype(int)
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
    df['monthly_spend'] = df['monthly_spend'].fillna(df['monthly_spend'].median())
    df['household_size'] = df['household_size'].fillna(df['household_size'].median()).astype(int)
    dfs["users"] = df
    print("   Cleaned users.csv")

    # Clean movies.csv
    df = dfs["movies"].copy()
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    df['added_to_platform'] = pd.to_datetime(df['added_to_platform'])
    df['genre_secondary'] = df['genre_secondary'].fillna('Unknown')
    df['imdb_rating'] = df['imdb_rating'].fillna(df['imdb_rating'].median())
    df['number_of_seasons'] = df['number_of_seasons'].fillna(0).astype(int)
    df['number_of_episodes'] = df['number_of_episodes'].fillna(0).astype(int)
    df['production_budget'] = df['production_budget'].fillna(0)
    df['box_office_revenue'] = df['box_office_revenue'].fillna(0)
    dfs["movies"] = df
    print("    Cleaned movies.csv")

    # Clean watch_history.csv
    df = dfs["watch_history"].copy()
    df = df.drop_duplicates(subset=['session_id'], keep='first')
    df['watch_date'] = pd.to_datetime(df['watch_date'])
    df['watch_duration_minutes'] = df['watch_duration_minutes'].fillna(df['watch_duration_minutes'].median())
    df['progress_percentage'] = df['progress_percentage'].fillna(df['progress_percentage'].median())
    df['user_rating'] = df['user_rating'].fillna(df['user_rating'].median())
    dfs["watch_history"] = df
    print("    Cleaned watch_history.csv")

    # Clean recommendation_logs.csv
    df = dfs["recommendation_logs"].copy()
    df = df.drop_duplicates(subset=['recommendation_id'], keep='first')
    df['recommendation_date'] = pd.to_datetime(df['recommendation_date'])
    df['recommendation_score'] = df['recommendation_score'].fillna(df['recommendation_score'].median())
    df['algorithm_version'] = df['algorithm_version'].fillna(df['algorithm_version'].mode()[0])
    dfs["recommendation_logs"] = df
    print("    Cleaned recommendation_logs.csv")

    # Clean search_logs.csv
    df = dfs["search_logs"].copy()
    df = df.drop_duplicates(subset=['search_id'], keep='first')
    df['search_date'] = pd.to_datetime(df['search_date'])
    df['clicked_result_position'] = df['clicked_result_position'].fillna(0)
    df['search_duration_seconds'] = df['search_duration_seconds'].fillna(df['search_duration_seconds'].median())
    dfs["search_logs"] = df
    print("    Cleaned search_logs.csv")

    # Clean reviews.csv
    df = dfs["reviews"].copy()
    df = df.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    df['review_date'] = pd.to_datetime(df['review_date'])
    df['helpful_votes'] = df['helpful_votes'].fillna(0).astype(int)
    df['total_votes'] = df['total_votes'].fillna(0).astype(int)
    df['review_text'] = df['review_text'].fillna('')
    df['sentiment_score'] = df['sentiment_score'].fillna(df['sentiment_score'].median())
    dfs["reviews"] = df
    print("   Cleaned reviews.csv")

    return dfs