"""
Meta-Learner Module
Combines CF and CBF scores with other features
Trains final classifier to predict click-through
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import VotingClassifier
import pickle
import os


def prepare_meta_features(df):
    """
    Prepare feature matrix for meta-learner
    Args:
        df: DataFrame with cf_score, cbf_score, and other engineered features
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Select features for meta-learner
    feature_cols = [
        # Stage 1 scores (from CF and CBF models)
        'cf_score',
        'cbf_score',

        # Interaction features
        'cf_cbf_interaction',
        'cf_search_interaction',
        'position_score',
        'high_rating_boost',

        # Recommendation context
        'rec_position',
        'time_of_day_encoded',

        # User features
        'age',
        'gender_encoded',
        'subscription_plan_encoded',
        'monthly_spend',
        'household_size',
        'is_active_int',
        'user_tenure_days',

        # User behavior
        'user_total_watches',
        'user_avg_watch_duration',
        'user_avg_completion',
        'user_total_downloads',
        'user_avg_rating',
        'user_total_reviews',

        # Movie features
        'duration_minutes',
        'imdb_rating',
        'is_netflix_original_int',

        # Match features
        'genre_match',
        'search_boost',
        'search_frequency'
    ]

    # Add dummy columns
    dummy_cols = [col for col in df.columns if col.startswith(('rec_type_', 'content_', 'device_'))]
    feature_cols.extend(dummy_cols)

    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"    WARNING: Missing columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]

    # Convert boolean columns to int (True=1, False=0)
    df_copy = df[feature_cols].copy()
    for col in feature_cols:
        if df_copy[col].dtype == bool:
            df_copy[col] = df_copy[col].astype(int)

    X = df_copy.values.astype(np.float64)
    y = df['target'].values.astype(int)

    return X, y, feature_cols


def train_logistic_regression(X_train, y_train, class_weights):
    """
    Train Logistic Regression with GridSearch
    """
    print("    Training Logistic Regression...")

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }

    lr = LogisticRegression(class_weight=class_weights, random_state=42)

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0)

    grid_search.fit(X_train, y_train)
    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def train_random_forest(X_train, y_train, class_weights):
    """
    Train Random Forest with GridSearch
    """
    print("    Training Random Forest...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }

    rf = RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0)

    grid_search.fit(X_train, y_train)

    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting with GridSearch
    """
    print("    Training Gradient Boosting...")

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [10, 20]
    }

    gb = GradientBoostingClassifier(random_state=42)

    # Compute sample weights for imbalanced data
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: class_weights_array[i] for i in range(len(classes))}
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    grid_search = GridSearchCV(
        gb,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0)

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def train_mlp_neural_network(X_train, y_train, class_weights):
    """
    Train MLP Neural Network with GridSearch
    """
    print("    Training MLP Neural Network...")

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500]}

    # Convert class_weights dict to string format for MLPClassifier
    mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)

    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: class_weights_array[i] for i in range(len(classes))}
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model on test set
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    print(f"\n      {model_name} Test Results:")
    print(f"        Accuracy:  {metrics['accuracy']:.4f}")
    print(f"        Precision: {metrics['precision']:.4f}")
    print(f"        Recall:    {metrics['recall']:.4f}")
    print(f"        F1-Score:  {metrics['f1']:.4f}")
    print(f"        ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def get_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from model
    """
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


def train_meta_learner(train_df, test_df):
    """
    Main function to train meta-learner models and compare them
    Returns:
        dict: Results including best model, metrics, and predictions
    """
    print("\n  Preparing meta-learning dataset...")

    # Prepare features
    X_train, y_train, feature_names = prepare_meta_features(train_df)
    X_test, y_test, _ = prepare_meta_features(test_df)

    print(f"    Training set: {X_train.shape}")
    print(f"    Test set: {X_test.shape}")
    print(f"    Number of features: {len(feature_names)}")

    # Check class distribution
    train_class_dist = pd.Series(y_train).value_counts()
    print(f"    Train class distribution: {train_class_dist.to_dict()}")
    print(f"    Class imbalance ratio: {train_class_dist[0] / train_class_dist[1]:.2f}:1")

    # Compute class weights automatically
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {classes[i]: class_weights_array[i] for i in range(len(classes))}
    print(f"    Computed class weights: {class_weights}")

    # Train and compare models
    print("\n  Training meta-learner models...")

    # Model 1: Logistic Regression
    print("\n  [Meta-Learner 1: Logistic Regression]")
    lr_model, lr_cv_score, lr_params = train_logistic_regression(X_train, y_train, class_weights)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Model 2: Random Forest
    print("\n  [Meta-Learner 2: Random Forest]")
    rf_model, rf_cv_score, rf_params = train_random_forest(X_train, y_train, class_weights)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Model 3: Gradient Boosting
    print("\n  [Meta-Learner 3: Gradient Boosting]")
    gb_model, gb_cv_score, gb_params = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    # Model 4: MLP Neural Network
    print("\n  [Meta-Learner 4: MLP Neural Network]")
    mlp_model, mlp_cv_score, mlp_params = train_mlp_neural_network(X_train, y_train, class_weights)
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, "MLP Neural Network")

    # Compare and select best model based on ROC-AUC
    models = {
        'Logistic Regression': (lr_model, lr_metrics, lr_params),
        'Random Forest': (rf_model, rf_metrics, rf_params),
        'Gradient Boosting': (gb_model, gb_metrics, gb_params),
        'MLP Neural Network': (mlp_model, mlp_metrics, mlp_params)
    }

    best_model_name = max(models.keys(), key=lambda k: models[k][1]['roc_auc'])
    best_model, best_metrics, best_params = models[best_model_name]

    print(f"\n  Winner: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")

    # Get feature importance
    print("\n  Extracting feature importance...")
    feature_importance = get_feature_importance(best_model, feature_names, best_model_name)

    if feature_importance is not None:
        print("\n  Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    # Ablation study: Test impact of removing key signals
    print("\n  Performing ablation study...")
    ablation_results = perform_ablation_study(train_df, test_df, best_model_name, class_weights)

    # Save best model
    os.makedirs('models', exist_ok=True)
    with open(f'models/best_meta_learner_{best_model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_params': best_params,
        'best_accuracy': best_metrics['accuracy'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_f1': best_metrics['f1'],
        'best_roc_auc': best_metrics['roc_auc'],
        'best_confusion_matrix': best_metrics['confusion_matrix'],
        'all_metrics': {
            'lr': lr_metrics,
            'rf': rf_metrics,
            'gb': gb_metrics,
            'mlp': mlp_metrics
        },
        'feature_importance': feature_importance,
        'ablation_results': ablation_results,
        'feature_names': feature_names
    }


def perform_ablation_study(train_df, test_df, best_model_name, class_weights):
    """
    Test impact of removing key features (cf_score, cbf_score, search_boost, interactions)
    """
    print("    Testing feature ablation...")

    ablation_configs = [
        ('Full Model', []),
        ('Without CF Score', ['cf_score', 'cf_cbf_interaction', 'cf_search_interaction']),
        ('Without CBF Score', ['cbf_score', 'cf_cbf_interaction']),
        ('Without Search Boost', ['search_boost', 'cf_search_interaction', 'search_frequency']),
        ('Without Interactions',
         ['cf_cbf_interaction', 'cf_search_interaction', 'position_score', 'high_rating_boost']),
        ('Without CF & CBF', ['cf_score', 'cbf_score', 'cf_cbf_interaction', 'cf_search_interaction'])
    ]

    results = {}

    for config_name, features_to_remove in ablation_configs:
        # Create modified datasets
        train_modified = train_df.copy()
        test_modified = test_df.copy()

        # Remove specified features (set to 0)
        for feat in features_to_remove:
            if feat in train_modified.columns:
                train_modified[feat] = 0
                test_modified[feat] = 0

        # Prepare features
        X_train, y_train, _ = prepare_meta_features(train_modified)
        X_test, y_test, _ = prepare_meta_features(test_modified)

        # Train simple model(logistic)
        model = LogisticRegression(C=1.0, class_weight=class_weights, max_iter=1000, random_state=42)

        model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results[config_name] = roc_auc
        print(f"      {config_name}: ROC-AUC = {roc_auc:.4f}")

    return results