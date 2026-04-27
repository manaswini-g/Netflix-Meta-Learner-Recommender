"""
Evaluation and Visualization Module
Creates comprehensive plots and displays them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def plot_cf_comparison(cf_results):
    """
    Plot comparison of CF models (SVD vs KNN)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['SVD', 'KNN']
    rmse_scores = [cf_results['svd_rmse'], cf_results['knn_rmse']]

    colors = ['#2ecc71' if model == cf_results['best_model_name'] else '#95a5a6' for model in models]

    bars = ax.bar(models, rmse_scores, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, score in zip(bars, rmse_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Collaborative Filtering Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rmse_scores) * 1.2)

    winner_idx = models.index(cf_results['best_model_name'])
    ax.annotate('WINNER',
                xy=(winner_idx, rmse_scores[winner_idx]),
                xytext=(winner_idx, rmse_scores[winner_idx] * 1.15),
                ha='center',
                fontsize=12,
                color='green',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.show()
    print("    Displayed: CF Model Comparison")


def plot_meta_learner_comparison(meta_results):
    """
    Plot comparison of meta-learner models
    """
    # Check which models we have
    available_models = []
    model_labels = []

    if 'lr' in meta_results['all_metrics']:
        available_models.append('lr')
        model_labels.append('Logistic\nRegression')
    if 'rf' in meta_results['all_metrics']:
        available_models.append('rf')
        model_labels.append('Random\nForest')
    if 'gb' in meta_results['all_metrics']:
        available_models.append('gb')
        model_labels.append('Gradient\nBoosting')
    if 'mlp' in meta_results['all_metrics']:
        available_models.append('mlp')
        model_labels.append('MLP\nNeural Net')

    if len(available_models) == 0:
        print("  No model metrics available for plotting")
        return

    metrics_data = {}
    for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        metric_key = metric_name.lower().replace('-', '_').replace(' ', '_')
        if metric_key == 'f1_score':
            metric_key = 'f1'
        if metric_key == 'roc_auc':
            metric_key = 'roc_auc'

        scores = []
        for model in available_models:
            scores.append(meta_results['all_metrics'][model][metric_key])
        metrics_data[metric_name] = scores

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric_name, scores) in enumerate(metrics_data.items()):
        ax = axes[idx]

        # Color the best model
        best_idx = scores.index(max(scores))
        colors = ['#2ecc71' if i == best_idx else '#3498db' for i in range(len(scores))]

        bars = ax.bar(model_labels, scores, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(f'Meta-Learner Model Comparison ({len(available_models)} Models)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    print("    Displayed: Meta-Learner Comparison")


def plot_confusion_matrix(meta_results):
    """
    Plot confusion matrix for best meta-learner
    """
    cm = meta_results['best_confusion_matrix']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Clicked', 'Clicked'],
                yticklabels=['Not Clicked', 'Clicked'],
                cbar_kws={'label': 'Count'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {meta_results["best_model_name"]}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    print("    Displayed: Confusion Matrix")


def plot_feature_importance(meta_results):
    """
    Plot top 15 most important features
    """
    if meta_results['feature_importance'] is None:
        print("  Feature importance not available for this model")
        return

    importance_df = meta_results['feature_importance'].head(15)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bar plot
    colors = ['#e74c3c' if 'cf_score' in feat or 'cbf_score' in feat or 'search_boost' in feat
              else '#3498db' for feat in importance_df['feature']]

    bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Feature Importance - {meta_results["best_model_name"]}',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Stage 1 Scores (CF/CBF/Search)'),
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='Other Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.show()
    print("    Displayed: Feature Importance")


def plot_ablation_study(meta_results):
    """
    Plot ablation study results
    """
    ablation = meta_results['ablation_results']

    configs = list(ablation.keys())
    scores = list(ablation.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color full model differently
    colors = ['#2ecc71' if 'Full' in config else '#e74c3c' for config in configs]

    bars = ax.barh(range(len(configs)), scores, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.,
                f'{score:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Impact of Removing Key Features',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(min(scores) * 0.95, max(scores) * 1.02)

    plt.tight_layout()
    plt.show()
    print("    Displayed: Ablation Study")


def plot_roc_curves(meta_results, test_df):
    """
    Plot ROC curves for all available meta-learner models
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Build models list
    models_data = []
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    color_idx = 0

    if 'lr' in meta_results['all_metrics']:
        models_data.append(('Logistic Regression', meta_results['all_metrics']['lr'], colors[color_idx]))
        color_idx += 1
    if 'rf' in meta_results['all_metrics']:
        models_data.append(('Random Forest', meta_results['all_metrics']['rf'], colors[color_idx]))
        color_idx += 1
    if 'gb' in meta_results['all_metrics']:
        models_data.append(('Gradient Boosting', meta_results['all_metrics']['gb'], colors[color_idx]))
        color_idx += 1
    if 'mlp' in meta_results['all_metrics']:
        models_data.append(('MLP Neural Network', meta_results['all_metrics']['mlp'], colors[color_idx]))
        color_idx += 1

    for model_name, metrics, color in models_data:
        y_true = test_df['target'].values
        y_proba = metrics['probabilities']

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        linestyle = '-' if model_name == meta_results['best_model_name'] else '--'
        linewidth = 3 if model_name == meta_results['best_model_name'] else 2

        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curves - Meta-Learner Comparison ({len(models_data)} Models)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("    Displayed: ROC Curves")


def create_model_summary_table(meta_results):
    """
    Create summary table with all model metrics
    """
    summary_data = []

    model_map = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'mlp': 'MLP Neural Network'
    }

    for model_key, model_name in model_map.items():
        if model_key in meta_results['all_metrics']:
            metrics = meta_results['all_metrics'][model_key]
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}"
            })

    summary_df = pd.DataFrame(summary_data)

    print("\n" + "=" * 100)
    print("MODEL SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Visual table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='left',
                     loc='center',
                     colColours=['#3498db'] * len(summary_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.show()
    print("    Displayed: Model Summary Table")


def print_detailed_metrics(cf_results, cbf_results, meta_results):
    """
    Print detailed metrics report to console
    """
    print("\n" + "=" * 80)
    print("DETAILED METRICS REPORT")
    print("=" * 80 + "\n")

    print("STAGE 1: COLLABORATIVE FILTERING")
    print("-" * 80)
    print(f"Best Model: {cf_results['best_model_name']}")
    print(f"RMSE: {cf_results['best_rmse']:.4f}")
    print(f"Best Parameters: {cf_results['best_params']}")
    print(f"SVD RMSE: {cf_results['svd_rmse']:.4f}")
    print(f"KNN RMSE: {cf_results['knn_rmse']:.4f}\n")

    print("STAGE 1: CONTENT-BASED FILTERING")
    print("-" * 80)
    print(f"Best Approach: {cbf_results['best_approach']}\n")

    print("STAGE 2: META-LEARNER")
    print("-" * 80)
    print(f"Best Model: {meta_results['best_model_name']}")
    print(f"Best Parameters: {meta_results['best_params']}\n")

    print("Test Set Performance:")
    print(f"  Accuracy:  {meta_results['best_accuracy']:.4f}")
    print(f"  Precision: {meta_results['best_precision']:.4f}")
    print(f"  Recall:    {meta_results['best_recall']:.4f}")
    print(f"  F1-Score:  {meta_results['best_f1']:.4f}")
    print(f"  ROC-AUC:   {meta_results['best_roc_auc']:.4f}\n")

    print("Confusion Matrix:")
    print(f"{meta_results['best_confusion_matrix']}\n")

    if meta_results['feature_importance'] is not None:
        print("Top 10 Most Important Features:")
        print(meta_results['feature_importance'].head(10).to_string(index=False))
        print("\n")

    print("Ablation Study Results:")
    for config, score in meta_results['ablation_results'].items():
        print(f"  {config}: {score:.4f}")

    print("\n" + "=" * 80 + "\n")


def evaluate_and_visualize(cf_results, cbf_results, meta_results, test_df):
    """
    Main function to create all visualizations and display them
    """
    print("\n  Creating and displaying visualizations...")

    print_detailed_metrics(cf_results, cbf_results, meta_results)
    plot_cf_comparison(cf_results)
    plot_meta_learner_comparison(meta_results)
    plot_confusion_matrix(meta_results)
    plot_feature_importance(meta_results)
    plot_ablation_study(meta_results)
    plot_roc_curves(meta_results, test_df)
    create_model_summary_table(meta_results)
