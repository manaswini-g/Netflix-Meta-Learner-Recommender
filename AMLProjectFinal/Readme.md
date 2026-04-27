# Meta-Learner Hybrid Recommendation System

## Project Overview

This project presents a three-stage machine learning pipeline designed to predict whether users will click on movie recommendations. The system integrates collaborative filtering, content-based filtering, and meta-learning to achieve 74% ROC-AUC in click-through prediction.

**Project Type**: End-to-end machine learning workflow prototype demonstrating systematic model development, feature engineering, and evaluation in a streaming service (Netflix) recommendation context.

---

## Problem Statement

**Business Problem**: Predict which recommended movies users are likely to click in order to improve user engagement and reduce decision fatigue.

**Machine Learning Task**: Binary classification (clicked vs. not clicked)

**Key Challenges**:

* Severe class imbalance (83% not clicked, 17% clicked)
* Data sparsity (85% missing explicit ratings)
* Cold start problem (new users/movies without interaction history)
* Multi-table feature engineering requirements

**Primary Metric**: ROC-AUC (selected due to class imbalance)

---

## Dataset Description

### Files Included (6 CSV files, 200,000+ total records)

#### 1. users.csv (10,000 records)

* `user_id` (str): Unique identifier, primary key
* `email` (str): User email address
* `first_name` (str): First name
* `last_name` (str): Last name
* `age` (int): Age in years, range 18-110, ~12% missing
* `gender` (str): Male/Female/Other/Non-binary, ~8% missing
* `country` (str): USA or Canada
* `state_province` (str): State or province name
* `city` (str): City name
* `subscription_plan` (str): Basic/Standard/Premium/Premium+
* `subscription_start_date` (datetime): Date subscription began
* `is_active` (bool): Currently active subscription
* `monthly_spend` (float): Monthly spending in USD, range $5-$50, ~10% missing
* `primary_device` (str): Mobile/Tablet/Smart TV/Laptop
* `household_size` (int): Number in household, range 1-8, ~15% missing
* `created_at` (datetime): Account creation timestamp

#### 2. movies.csv (1,000 records)

* `movie_id` (str): Unique identifier, primary key
* `title` (str): Movie/show title
* `content_type` (str): Movie/TV Series/Documentary/Limited Series/Stand-up Comedy
* `genre_primary` (str): Primary genre
* `genre_secondary` (str): Secondary genre, ~40% missing
* `release_year` (int): Year released
* `duration_minutes` (float): Length in minutes
* `rating` (str): Content rating
* `language` (str): Primary language
* `country_of_origin` (str): Country produced
* `imdb_rating` (float): IMDB rating, scale 1-10, ~15% missing
* `production_budget` (float): Budget in USD, ~20% missing
* `box_office_revenue` (float): Revenue in USD, ~25% missing
* `number_of_seasons` (int): For TV series, 0 for movies
* `number_of_episodes` (int): Total episodes, 0 for movies
* `is_netflix_original` (bool): Original production flag
* `added_to_platform` (datetime): Date added to platform
* `content_warning` (bool): Mature content warning indicator

#### 3. watch_history.csv (100,000 records)

* `session_id` (str): Unique identifier, primary key
* `user_id` (str): Foreign key to users table
* `movie_id` (str): Foreign key to movies table
* `watch_date` (datetime): Viewing session timestamp
* `device_type` (str): Mobile/Tablet/Smart TV/Laptop
* `watch_duration_minutes` (float): Actual watch time, ~8% missing
* `progress_percentage` (float): Completion percentage 0-100, ~12% missing
* `action` (str): started/completed/stopped/paused
* `quality` (str): HD/4K/SD/Ultra HD
* `location_country` (str): USA/Canada
* `is_download` (bool): Offline download flag
* `user_rating` (float): Optional 1-5 star rating, ~85% missing

#### 4. recommendation_logs.csv (50,000 records) ← TARGET VARIABLE

* `recommendation_id` (str): Unique identifier, primary key
* `user_id` (str): Foreign key to users table
* `movie_id` (str): Foreign key to movies table
* `recommendation_date` (datetime): Recommendation timestamp
* `recommendation_type` (str): genre_based/personalized/similar_users/trending/new_releases
* `recommendation_score` (float): Algorithm confidence score 0-1
* `was_clicked` (bool): **TARGET VARIABLE** - Did user click? (True/False)
* `position_in_list` (int): Position in recommendation list, range 1-20
* `device_type` (str): Mobile/Tablet/Smart TV/Laptop
* `time_of_day` (str): morning/afternoon/evening/night
* `algorithm_version` (str): v1.2/v1.3/v1.4

#### 5. search_logs.csv (25,000 records)

* `search_id` (str): Unique identifier, primary key
* `user_id` (str): Foreign key to users table
* `search_query` (str): User search text
* `search_date` (datetime): Search timestamp
* `results_returned` (int): Number of results shown
* `clicked_result_position` (float): Which result was clicked, ~51% missing
* `device_type` (str): Mobile/Tablet/Smart TV/Laptop
* `search_duration_seconds` (float): Time spent searching, ~5% missing
* `had_typo` (bool): Search contained spelling error
* `used_filters` (bool): Applied genre/year/rating filters
* `location_country` (str): USA/Canada

#### 6. reviews.csv (15,000 records)

* `review_id` (str): Unique identifier, primary key
* `user_id` (str): Foreign key to users table
* `movie_id` (str): Foreign key to movies table
* `rating` (int): Explicit 1-5 star rating
* `review_date` (datetime): Review timestamp
* `device_type` (str): Mobile/Tablet/Smart TV/Laptop
* `is_verified_watch` (bool): Verified watch indicator
* `helpful_votes` (int): Number of helpful votes, ~12% missing
* `total_votes` (int): Total votes received, ~12% missing
* `review_text` (str): Text review, ~5% missing
* `sentiment` (str): positive/negative/neutral
* `sentiment_score` (float): Sentiment score 0-1, ~8% missing

---

## Project Structure

```text
AMLProject/
│
├── datasets/                # All 6 CSV files (extracted from Kaggle Netflix dataset)
│   ├── users.csv
│   ├── movies.csv
│   ├── watch_history.csv
│   ├── recommendation_logs.csv
│   ├── search_logs.csv
│   └── reviews.csv
│
├── main.py
├── data_loader.py
├── feature_engineering.py
├── collaborative_filtering.py
├── content_based_filtering.py
├── meta_learner.py
├── evaluation.py
│
├── models/
│   ├── best_cf_model_knn.pkl
│   ├── best_cf_model_svd.pkl
│   ├── best_cbf_model.pkl
│   ├── best_meta_learner_logistic_regression.pkl
│   ├── best_meta_learner_random_forest.pkl
│   ├── best_meta_learner_gradient_boosting.pkl
│   └── best_meta_learner_mlp_neural_network.pkl
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Installation Steps

```bash
# 1. Clone or download project
cd AMLProjectFinal

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify datasets are in place
# Ensure all 6 CSV files are in the datasets/ folder
```

### Requirements

```text
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
```

---

## Running the Project

The `main.py` script serves as the primary execution entry point for the full pipeline.

### Full Pipeline Execution

```bash
python main.py
```

### Pipeline Workflow

Execution performs the following:

1. Loads and cleans all 6 CSV files
2. Engineers 43 features from multiple tables
3. Creates temporal train/test split (80/20)
4. Trains collaborative filtering models (SVD vs KNN)
5. Trains content-based filtering models (3 approaches)
6. Trains 4 meta-learner models with GridSearchCV
7. Performs ablation study (feature contribution analysis)
8. Generates 7 interactive visualizations
9. Saves all trained models to the `models/` directory

### Output

* Console logs with detailed metrics at each stage
* 7 matplotlib windows:

  1. CF Model Comparison (SVD vs KNN)
  2. Meta-Learner Comparison (4 models)
  3. Confusion Matrix (best model)
  4. Feature Importance (top 15 features)
  5. Ablation Study (feature contribution)
  6. ROC Curves (all 4 models)
  7. Model Summary Table
* Saved models in `models/` directory (7 .pkl files)

---

# Methodology

## Stage 0: Feature Engineering (43 Features from 6 Tables)

### User Behavioral Features (12 features)

Includes:

* `user_total_watches`
* `user_avg_watch_duration`
* `user_avg_completion`
* `user_total_downloads`
* `user_avg_rating`
* `user_total_reviews`
* `user_tenure_days`
* `user_engagement_score`
* `days_since_last_watch`
* `age`, `monthly_spend`, `household_size`

### Content Features (8 features)

Includes:

* `imdb_rating`
* `duration_minutes`
* `is_netflix_original_int`
* `movie_popularity_score`
* `content_type` dummies

### Match Features (6 features)

Includes:

* `genre_match`
* `search_boost`
* `search_frequency`
* `high_rating_boost`

### Contextual Features (9 features)

Includes:

* `rec_position`
* `position_score`
* `time_of_day_encoded`
* `device_type` dummies
* `mobile_evening`
* `tv_weekend`

### Recommendation Metadata (8 features)

Includes:

* `recommendation_type` dummies
* `algo_recommendation_score`

**Total**: 43 numerical features (all categorical variables one-hot encoded)

---

## Stage 1a: Collaborative Filtering

**Goal**: Generate user preference scores from interaction patterns.

**Input**: User-item matrix (10,000 users × 1,000 movies)
**Sparsity**: 98.86% (only 1.14% of cells contain ratings)

### Rating Construction

* Explicit ratings: 14,993 from reviews.csv
* Implicit ratings: 99,349 from watch_history.csv

  * Formula: `(completion% / 100) × 4 + 1`
* **Total**: 114,342 ratings

### Algorithms Compared

#### 1. TruncatedSVD (Matrix Factorization)

* Hyperparameters tested: `n_components` = 50, 100, 150
* Best: n_components=150
* RMSE: 2.2708

#### 2. KNN (Nearest Neighbors) ★ Best Performing Model

* Hyperparameters tested: `k` = 20, 40, 60
* Best: k=20
* RMSE: 0.6441
* Metric: Cosine similarity

**Selected Model**: KNN with k=20

**Output**: `cf_score` column (0-1 scale)

**Cold Start Handling**: Users/movies without history receive score=0.5 (neutral)

---

## Stage 1b: Content-Based Filtering

**Goal**: Generate content similarity scores using movie metadata.

### Approaches Compared

1. **Genre-only**

* Correlation with clicks: 0.0382

2. **Multi-feature**

* Correlation with clicks: 0.0394

3. **Hybrid (Text + Numerical)** Best Performing Approach

* TF-IDF (70% weight) + normalized IMDB/duration (30% weight)
* Correlation with clicks: 0.0655

**Selected Approach**: Hybrid approach (50% better correlation than genre-only)

**Method**:

* TF-IDF vectorization (max 100 features)
* StandardScaler for numerical features
* Cosine similarity between recommended movie and user watch history

**Output**: `cbf_score` column (0-1 scale)

**Cold Start Handling**: Users without history receive genre-based similarity only.

---

## Stage 2: Meta-Learner (Final Classifier)

**Goal**: Predict click probability by combining CF/CBF scores with engineered features.

**Input**: 43 features (cf_score, cbf_score, + 41 engineered features)

### Class Distribution

* Overall synthetic click rate: 16.1% (class distribution: {0: 16,771, 1: 3,229})
* Train: 13,365 not clicked, 2,635 clicked
* Imbalance ratio: ≈5.07 : 1
* Test: 4,000 samples (3,406 not clicked, 594 clicked)

### Class Imbalance Handling

* Balanced class weights: {0: 0.60, 1: 3.04}
* Applied to all models
* Stratified sampling used

### Algorithms Compared (all with GridSearchCV, 3-fold CV)

#### 1. Logistic Regression ★ Selected Final Model

**Best Parameters**: C=10.0, L2 penalty
**CV ROC-AUC**: 0.7403
**Test ROC-AUC**: 0.7400
**Test Accuracy**: 66.93%
**Precision**: 26.65%, **Recall**: 70.03%, **F1**: 0.3861

#### 2. Random Forest

**CV ROC-AUC**: 0.7441
**Test ROC-AUC**: 0.7391
**Test Accuracy**: 73.45%
**Precision**: 30.82%, **Recall**: 63.30%, **F1**: 0.4146

#### 3. Gradient Boosting

**CV ROC-AUC**: 0.7420
**Test ROC-AUC**: 0.7387
**Test Accuracy**: 71.85%
**Precision**: 29.66%, **Recall**: 65.32%, **F1**: 0.4080

#### 4. MLP Neural Network

**CV ROC-AUC**: 0.7200
**Test ROC-AUC**: 0.6954
**Test Accuracy**: 84.88% (majority class driven)
**Precision**: 21.05%, **Recall**: 0.67%, **F1**: 0.0131

---

## Synthetic Data & Pattern Engineering

### Context

This project uses the **Netflix User Behaviour** dataset extracted from Kaggle. Analysis indicated that the synthetic dataset’s original `was_clicked` target variable was randomly assigned without learnable patterns. Initial models achieved ~50% ROC-AUC (random baseline), indicating no natural predictive signal.

### Approach: Research-Based Pattern Engineering

To validate the ML pipeline architecture and demonstrate workflow rigor, realistic click behavior patterns were engineered using research-based assumptions and domain-informed logic.

#### Pattern 1: Position Bias (Exponential Decay)

**Implementation**: `click_probability = 0.70 × exp(-0.25 × position)`

#### Pattern 2: Genre-Content Alignment

**Implementation**: +8% boost when recommended genre matches user's most-watched genre

#### Pattern 3: Search Intent Signal

**Implementation**: +25% boost if user searched related genre/title within 30 days

#### Pattern 4: Contextual Interactions

**Implementation**: Mobile+Evening → +8%, SmartTV+Weekend → +10%

#### Pattern 5: Quality-Engagement Multiplier

**Implementation**: High-engagement users × High-IMDB ratings → Multiplicative boost

### Pattern Engineering Code Location

See `feature_engineering.py`, function `engineer_click_patterns_hybrid()` (lines 450-650)

### What This Project Validates

* End-to-end ML pipeline construction
* Multi-table feature engineering
* Handling severe class imbalance
* Multi-stage learning architecture
* Systematic model comparison
* Hyperparameter tuning with GridSearchCV
* Proper temporal train/test splitting
* Ablation study methodology
* Comprehensive evaluation framework

### Results Interpretation

* **ROC-AUC 0.74**: Models learned engineered patterns successfully
* **Position bias dominance**: Supports realism of pattern engineering
* **CF/CBF weakness**: Expected from random original data
* **Logistic Regression performance**: Linear model sufficient for engineered signals

### Transition to Production

To deploy with real click data:

1. Remove pattern engineering module (`engineer_click_patterns_hybrid()`)
2. Retain feature engineering components
3. Retrain CF/CBF models on real interaction data
4. Re-tune meta-learner hyperparameters
5. Conduct A/B testing against baseline recommender system

---

# Results

## Best Model: Logistic Regression

| Metric      | Value      | Interpretation                              |
| ----------- | ---------- | ------------------------------------------- |
| **ROC-AUC** | **0.7400** | Strong discrimination for click prediction  |
| Accuracy    | 66.93%     | Correctly classified 2,677 of 4,000 samples |
| Precision   | 26.65%     | 1 in 4 positive predictions correct         |
| Recall      | 70.03%     | Captured 70% of actual clicks               |
| F1-Score    | 0.3861     | Harmonic mean of precision and recall       |

## Confusion Matrix (Test Set: 4,000 samples)

```text
                Predicted
Actual      Not Clicked  |  Clicked
-----------------------------------------
Not Clicked   2,261 (TN)  |  1,145 (FP)
Clicked         178 (FN)  |    416 (TP)
```

### Business Interpretation

* **True Positives (416)**: Captured recommendation engagement opportunities
* **False Negatives (178)**: Missed 31% of interested users
* **False Positives (1,145)**: Inefficient recommendation placements
* **True Negatives (2,261)**: Correctly filtered low-interest content

---

## Feature Importance (Top 10)

| Rank | Feature                 | Importance | Category            |
| ---- | ----------------------- | ---------- | ------------------- |
| 1    | genre_match             | 0.7382     | Match Signal        |
| 2    | position_score          | 0.6773     | Position Bias       |
| 3    | content_Stand-up Comedy | 0.1548     | Content Type        |
| 4    | cf_score                | 0.1456     | CF Signal           |
| 5    | rec_position            | 0.1424     | Position Bias       |
| 6    | imdb_rating             | 0.1290     | Quality Signal      |
| 7    | device_Mobile           | 0.1219     | Context             |
| 8    | rec_type_personalized   | 0.1096     | Recommendation Type |
| 9    | is_active_int           | 0.1071     | User Status         |
| 10   | content_Movie           | 0.1028     | Content Type        |

### Key Insights

* Position bias is the dominant signal
* Genre matching is the strongest predictor
* CF score contributes measurable predictive value
* Search intent provides meaningful behavioral signal

---

## Ablation Study Results

| Configuration        | ROC-AUC | Change   | Interpretation                 |
| -------------------- | ------- | -------- | ------------------------------ |
| Full Model           | 0.7392  | Baseline | All 43 features                |
| Without CF Score     | 0.7398  | +0.0006  | CF adds slight noise           |
| Without CBF Score    | 0.7402  | +0.0010  | CBF adds slight noise          |
| Without Search Boost | 0.7390  | -0.0002  | Search contributes slightly    |
| Without Interactions | 0.7397  | +0.0005  | Interactions largely neutral   |
| Without CF & CBF     | 0.7401  | +0.0009  | Combined scores slightly noisy |

### Key Finding

Removing CF/CBF scores marginally improved performance.

**Interpretation**:

* CF/CBF were generated from weak synthetic signals
* Real behavioral data is expected to improve their value
* Feature utility should be empirically validated rather than assumed

---

## Model Comparison Summary

| Model                 | ROC-AUC | Accuracy | F1-Score | Pros                  | Cons                          |
| --------------------- | ------- | -------- | -------- | --------------------- | ----------------------------- |
| Logistic Regression ★ | 0.7400  | 66.93%   | 0.3861   | Interpretable, fast   | Lower accuracy                |
| Random Forest         | 0.7391  | 73.45%   | 0.4146   | Strong F1             | Slightly lower AUC            |
| Gradient Boosting     | 0.7387  | 71.85%   | 0.4080   | Balanced performance  | Slower training               |
| MLP Neural Network    | 0.6954  | 84.88%   | 0.0131   | High nominal accuracy | Poor minority class detection |

### Why MLP Underperformed

* Only 0.67% recall for clicks
* Accuracy driven by majority class prediction
* Imbalanced problems require more careful neural network calibration

---

# Key Learnings & Takeaways

## 1. Feature Engineering > Model Complexity

* Carefully engineered features outperformed more complex deep learning approaches
* Domain-informed features were primary drivers of performance
* Simpler features offer interpretability and production maintainability

## 2. Simpler Models Can Outperform Complex Models

* Logistic Regression outperformed more sophisticated alternatives
* Simpler models may generalize better under structured signals

## 3. Validate Assumptions Empirically

* Ablation study revealed unexpected weakness in CF/CBF features
* Additional features do not guarantee better performance

## 4. Class Imbalance Requires Dedicated Treatment

* Accuracy alone is insufficient
* ROC-AUC, precision, recall, and F1 provide a fuller evaluation
* Class weighting and stratified sampling were critical

## 5. Position Bias is a Strong Behavioral Signal

* Top recommendation positions have disproportionate impact
* Placement optimization has direct business implications

## 6. Real-World Data Requires Robust Design

* Missingness and sparsity required fallback strategies
* Production systems must accommodate imperfect data conditions

---

## Technical Specifications

### Reproducibility

* All random seeds set to 42 (`random_state=42`)
* Stratified sampling preserves class distribution
* Temporal split (80/20) prevents data leakage
* GridSearchCV with fixed CV splits

---

# File Descriptions

## Core Scripts (7 files)

**main.py**
Orchestrates the full pipeline execution.

**data_loader.py**
Loads 6 CSV files, cleans missing values, converts dates, removes duplicates.

**feature_engineering.py**
Creates 43 features from 6 tables, including synthetic pattern engineering.

**collaborative_filtering.py**
Trains SVD and KNN models and generates collaborative filtering scores.

**content_based_filtering.py**
Trains TF-IDF models and compares 3 content-based approaches.

**meta_learner.py**
Trains 4 classification models using GridSearchCV and performs ablation studies.

**evaluation.py**
Generates visualizations and reports evaluation metrics.

## Generated Files (Auto-Created During Execution)

**models/ directory** (7 .pkl files):

* `best_cf_model_knn.pkl` - KNN collaborative filtering model
* `best_cf_model_svd.pkl` - SVD collaborative filtering model
* `best_cbf_model.pkl` - Hybrid content-based filtering model
* `best_meta_learner_logistic_regression.pkl` - Selected meta-learner
* `best_meta_learner_random_forest.pkl` - Random Forest meta-learner
* `best_meta_learner_gradient_boosting.pkl` - Gradient Boosting meta-learner
* `best_meta_learner_mlp_neural_network.pkl` - MLP meta-learner

---

This repository is structured to emphasize reproducibility, modularity, and end-to-end machine learning workflow design, while preserving all original methodology, variables, code references, and reported results.
