"""
Meta-Learner Hybrid Recommendation System

Main execution point  - Run this file to execute the entire pipeline and all datastes and other scripts will
automatically be loaded and executed.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import load_and_clean_data
from feature_engineering import create_features
from collaborative_filtering import train_collaborative_models
from content_based_filtering import train_content_based_model
from meta_learner import train_meta_learner
from evaluation import evaluate_and_visualize


def main():
    """
    Main pipeline for the recommendation system
    """
    print("=" * 80)
    print("META-LEARNER HYBRID RECOMMENDATION SYSTEM")
    print("=" * 80)

    # Load and Clean Data
    print("\n Loading and cleaning datasets...")
    datasets = load_and_clean_data()
    print(" Data loaded & cleaned successfully!")

    # Feature Engineering
    print("\n Engineering features from multiple tables...")
    feature_df = create_features(datasets)
    print(f" Created feature dataset with shape: {feature_df.shape}")

    # Temporal Train/Test Split
    print("\n Creating temporal train/test split...")
    feature_df['recommendation_date'] = pd.to_datetime(feature_df['recommendation_date'])
    feature_df = feature_df.sort_values('recommendation_date')

    # Use 80% for training (earlier dates), 20% for testing (later dates)
    split_idx = int(len(feature_df) * 0.8)
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    print(f"  Training set: {len(train_df)} samples ({train_df['recommendation_date'].min()} to {train_df['recommendation_date'].max()})")
    print(f"  Test set: {len(test_df)} samples ({test_df['recommendation_date'].min()} to {test_df['recommendation_date'].max()})")

    # STEP 4: Train Collaborative Filtering Models
    print("\n Training Collaborative Filtering models (SVD vs KNN)...")
    cf_results = train_collaborative_models(datasets, train_df, test_df)
    print(f" Best CF Model: {cf_results['best_model_name']} with RMSE: {cf_results['best_rmse']:.4f}")

    # Add CF scores to our feature dataframes
    train_df['cf_score'] = cf_results['train_predictions']
    test_df['cf_score'] = cf_results['test_predictions']

    # Content-Based Filtering
    print("\nTraining Content-Based Filtering models...")
    cbf_results = train_content_based_model(datasets, train_df, test_df)
    print(f" Best CBF Approach: {cbf_results['best_approach']}")

    # Adding CBF scores to our feature dataframes
    train_df['cbf_score'] = cbf_results['train_scores']
    test_df['cbf_score'] = cbf_results['test_scores']

    # Interaction Features
    print("\n Creating interaction features...")

    # Interaction 1: CF × CBF (do both signals agree?)
    train_df['cf_cbf_interaction'] = train_df['cf_score'] * train_df['cbf_score']
    test_df['cf_cbf_interaction'] = test_df['cf_score'] * test_df['cbf_score']

    # Interaction 2: CF × Search Boost (is CF predicting something user searched for?)
    train_df['cf_search_interaction'] = train_df['cf_score'] * train_df['search_boost']
    test_df['cf_search_interaction'] = test_df['cf_score'] * test_df['search_boost']

    # Interaction 3: Position Score (higher weight to top positions)
    train_df['position_score'] = 1.0 / (train_df['rec_position'] + 1)
    test_df['position_score'] = 1.0 / (test_df['rec_position'] + 1)

    # Interaction 4: High rating boost (IMDB > 7.5)
    train_df['high_rating_boost'] = (train_df['imdb_rating'] > 7.5).astype(int)
    test_df['high_rating_boost'] = (test_df['imdb_rating'] > 7.5).astype(int)

    print("  Created interaction features")
    print(f"    cf_cbf_interaction range: [{train_df['cf_cbf_interaction'].min():.3f}, {train_df['cf_cbf_interaction'].max():.3f}]")
    print(f"    position_score range: [{train_df['position_score'].min():.3f}, {train_df['position_score'].max():.3f}]")

    #  Meta-Learner
    print("\n Training Meta-Learner (Final Stage)...")
    meta_results = train_meta_learner(train_df, test_df)
    print(f" Best Meta-Learner: {meta_results['best_model_name']}")
    print(f"  Test ROC-AUC: {meta_results['best_roc_auc']:.4f}")
    print(f"  Test Accuracy: {meta_results['best_accuracy']:.4f}")

    #  Comprehensive Evaluation and Visualization
    print("\nDisplaying evaluation results and visualizations...")
    evaluate_and_visualize(cf_results, cbf_results, meta_results, test_df)



    print("=" * 80)
    print("\nKey Results:")
    print(f"  Best CF Model: {cf_results['best_model_name']}")
    print(f"  Best CBF Approach: {cbf_results['best_approach']}")
    print(f"  Best Meta-Learner: {meta_results['best_model_name']}")
    print(f"  Final Test ROC-AUC: {meta_results['best_roc_auc']:.4f}")


if __name__ == "__main__":
    main()