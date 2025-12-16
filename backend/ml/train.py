"""
Random Forest Model Training for Food Recommendation System
Author: NutriSolve ML Team
Date: December 2025

Theoretical Foundation:
Random Forest Classifier - Bagging ensemble of N decision trees
Mathematical Formulation:
- Each tree T_i builds via recursive splits minimizing Gini impurity:
  G(node_m) = 1 - Σ(k=1 to K) p_mk²
  where p_mk = proportion of class k in node m, K = number of classes
  
- Aggregate prediction for classification:
  P(y=1|x) = (1/N) Σ(i=1 to N) P_i(y=1|x)
  where P_i is the prediction from tree T_i's leaf node
  
- Final classification: majority vote across all trees
  ŷ = argmax_k Σ(i=1 to N) I(T_i(x) = k)

Why Random Forest:
1. Handles non-linear relationships (nutrition data interactions)
2. Robust to outliers and noise (variable nutrient levels)
3. Bagging reduces variance: σ_ensemble ≈ σ_tree / √N
4. Feature importance via Gini decrease (interpretability)
5. No hyperparameter sensitivity (vs SVM kernel tuning)

Small Dataset Adaptations (205 samples, 34.6% minority):
- Conservative hyperparameters to prevent overfitting
- max_depth ≈ log2(n_samples) = 7-8
- min_samples_split ≈ n_samples/40 = 5
- min_samples_leaf ≈ n_samples/100 = 2-3
- No class_weight='balanced' (34% is naturally balanced)

Hyperparameter Tuning:
GridSearchCV with 5-fold cross-validation
Scoring: f1_macro (balances precision and recall)
Conservative search space for 205 samples
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define paths (must match preprocess.py)
BASE_DIR = Path(__file__).parent
ML_DIR = Path(__file__).parent

def load_training_data():
    """
    Load preprocessed training and test data from preprocess.py outputs
    All preprocessing (scaling, encoding, feature selection) already done
    Returns: X_train, y_train, X_test, y_test, feature_names
    """
    print("[Train] Loading preprocessed data...")
    
    train_df = pd.read_csv(ML_DIR / 'train_data.csv')
    test_df = pd.read_csv(ML_DIR / 'test_data.csv')
    
    # Load feature metadata from preprocess.py
    with open(ML_DIR / 'feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    # Use selected features from preprocessing (already selected by SelectKBest)
    feature_names = feature_info['selected_features']
    
    print(f"  - Loaded {len(feature_names)} selected features from preprocessing")
    print(f"  - Features: {feature_names}")
    
    # Separate features and target (all preprocessing already applied)
    X_train = train_df[feature_names].values
    y_train = train_df['fit'].values
    X_test = test_df[feature_names].values
    y_test = test_df['fit'].values
    
    print(f"\n  - Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  - Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"  - Train class distribution: fit=1 ({y_train.sum()}, {y_train.mean()*100:.1f}%), fit=0 ({len(y_train)-y_train.sum()}, {(1-y_train.mean())*100:.1f}%)")
    print(f"  - Test class distribution: fit=1 ({y_test.sum()}, {y_test.mean()*100:.1f}%), fit=0 ({len(y_test)-y_test.sum()}, {(1-y_test.mean())*100:.1f}%)")
    
    return X_train, y_train, X_test, y_test, feature_names

def perform_hyperparameter_tuning(X_train, y_train):
    """
    Conservative hyperparameter tuning for small dataset (205 samples)
    
    Search Strategy:
    - 5-fold cross-validation (preserves class distribution)
    - Scoring: f1_macro (balanced precision/recall)
    - Conservative parameters to prevent overfitting on 205 samples
    
    Parameter Grid (Small Dataset Rules):
    - n_estimators: [50, 100]
      Fewer trees for small data, diminishing returns beyond 100
    
    - max_depth: [3, 5, 7]
      Based on log2(n_samples) ≈ log2(205) ≈ 7.7
      Conservative to prevent memorization
    
    - min_samples_split: [5, 10, 15]
      Approx n_samples/40 = 205/40 ≈ 5
      Forces trees to generalize, not overfit
    
    - min_samples_leaf: [3, 5, 7]
      Approx n_samples/100 = 205/100 ≈ 2-3
      Ensures leaf nodes have sufficient support
    
    No class_weight: 34.6% minority is naturally balanced
    """
    print("\n[Train] Starting hyperparameter tuning (small dataset strategy)...")
    print("  - Method: GridSearchCV")
    print("  - Cross-validation: 5-fold")
    print("  - Scoring: f1_macro")
    print(f"  - Dataset size: {len(X_train)} samples (conservative parameters)")
    
    # Conservative parameter grid for 205 samples
    param_grid = {
        'n_estimators': [50, 100],              # Fewer trees for small data
        'max_depth': [3, 5, 7],                 # log2(205) ≈ 7-8
        'min_samples_split': [5, 10, 15],       # n_samples/40 ≈ 5
        'min_samples_leaf': [3, 5, 7],          # n_samples/100 ≈ 2-3
        'random_state': [RANDOM_STATE]          # Reproducibility
    }
    
    n_combinations = (len(param_grid['n_estimators']) * 
                      len(param_grid['max_depth']) * 
                      len(param_grid['min_samples_split']) * 
                      len(param_grid['min_samples_leaf']))
    print(f"  - Parameter grid: {n_combinations} combinations")
    print(f"  - No class_weight (minority class 34.6% is naturally balanced)")
    
    # Initialize base model
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # GridSearchCV: Exhaustive search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,                      # 5-fold cross-validation
        scoring='f1_macro',        # Evaluation metric
        n_jobs=1,                  # Single job for small dataset
        verbose=1,                 # Progress updates
        return_train_score=True    # Track train scores for overfitting detection
    )
    
    # Fit grid search
    print("\n[Train] Fitting GridSearchCV (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    # Extract best parameters and scores
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n[Train] Hyperparameter tuning complete!")
    print(f"  - Best cross-val f1_macro: {best_score:.4f}")
    print(f"  - Best parameters:")
    for param, value in best_params.items():
        if param != 'random_state':
            print(f"    • {param}: {value}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Display top 5 configurations
    print("\n[Train] Top 5 configurations by f1_macro:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 
         'param_min_samples_leaf', 'mean_test_score', 'std_test_score']
    ]
    print(top_5.to_string(index=False))
    
    return best_model, best_params

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation on training and test sets
    
    Metrics:
    1. Accuracy: (TP + TN) / Total
    2. Precision: TP / (TP + FP) - "How many predicted positives are correct?"
    3. Recall: TP / (TP + FN) - "How many actual positives did we find?"
    4. F1-score: 2 · (Precision · Recall) / (Precision + Recall) - Harmonic mean
    5. ROC-AUC: Area under Receiver Operating Characteristic curve
    6. Confusion Matrix: Visualize TP, TN, FP, FN
    
    Why these metrics:
    - Accuracy insufficient for imbalanced data (can be high by predicting majority)
    - F1 balances precision/recall (critical for both false positives and false negatives)
    - ROC-AUC measures ranking quality (important for top-k recommendations)
    """
    print("\n[Train] Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Training set metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Test set metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Print summary table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'Training':<20} {'Test':<20} {'Difference':<10}")
    print("-"*70)
    print(f"{'Accuracy':<20} {train_acc:>19.4f} {test_acc:>19.4f} {abs(train_acc-test_acc):>9.4f}")
    print(f"{'F1-score (macro)':<20} {train_f1:>19.4f} {test_f1:>19.4f} {abs(train_f1-test_f1):>9.4f}")
    print(f"{'Precision (macro)':<20} {train_precision:>19.4f} {test_precision:>19.4f} {abs(train_precision-test_precision):>9.4f}")
    print(f"{'Recall (macro)':<20} {train_recall:>19.4f} {test_recall:>19.4f} {abs(train_recall-test_recall):>9.4f}")
    print(f"{'ROC-AUC':<20} {train_auc:>19.4f} {test_auc:>19.4f} {abs(train_auc-test_auc):>9.4f}")
    print("="*70)
    
    # Check for overfitting (stricter threshold for small dataset)
    f1_gap = train_f1 - test_f1
    if f1_gap > 0.07:
        print(f"\nWARNING: Potential overfitting detected (train-test F1 gap = {f1_gap:.4f} > 0.07)")
        print("   Small dataset (205 samples) requires gap < 0.07 for good generalization")
    else:
        print(f"\nGood generalization (train-test F1 gap = {f1_gap:.4f} < 0.07)")
    
    # Detailed classification report (per-class metrics)
    print("\n[Train] Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Unfit (0)', 'Fit (1)'],
                                digits=4))
    
    # Confusion matrix
    print("[Train] Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                Predicted Unfit  Predicted Fit")
    print(f"Actual Unfit    {cm[0,0]:>15}  {cm[0,1]:>13}")
    print(f"Actual Fit      {cm[1,0]:>15}  {cm[1,1]:>13}")
    
    # Feature importances (top 10)
    print("\n[Train] Top 10 Feature Importances (Gini decrease):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Percentage'}")
    print("-"*70)
    for rank, idx in enumerate(indices, 1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        importance = importances[idx]
        percentage = importance * 100
        print(f"{rank:<6} {feat_name:<30} {importance:>11.6f} {percentage:>10.2f}%")
    
    # Return metrics dictionary
    metrics = {
        'train': {
            'accuracy': float(train_acc),
            'f1_macro': float(train_f1),
            'precision_macro': float(train_precision),
            'recall_macro': float(train_recall),
            'roc_auc': float(train_auc)
        },
        'test': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1),
            'precision_macro': float(test_precision),
            'recall_macro': float(test_recall),
            'roc_auc': float(test_auc)
        },
        'confusion_matrix': cm.tolist(),
        'feature_importances': {
            feature_names[i] if i < len(feature_names) else f"Feature_{i}": float(importances[i])
            for i in range(len(importances))
        }
    }
    
    return metrics

def train_model():
    """
    Main training pipeline for small dataset (205 samples)
    
    Steps:
    1. Load preprocessed data (scaling, encoding, feature selection already done)
    2. Perform conservative hyperparameter tuning via GridSearchCV
    3. Train final model with best parameters
    4. Evaluate on train and test sets
    5. Save trained model and metrics
    
    No preprocessing in this file - all done in preprocess.py
    
    Output:
    - rf_model.pkl: Trained Random Forest model (for predict.py)
    - training_metrics.json: Performance metrics (for documentation)
    """
    print("\n" + "="*70)
    print("FOOD RECOMMENDATION ML PIPELINE - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Step 1: Load data
    X_train, y_train, X_test, y_test, feature_names = load_training_data()
    
    # Step 2: Hyperparameter tuning
    best_model, best_params = perform_hyperparameter_tuning(X_train, y_train)
    
    # Step 3: Evaluate model
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, feature_names)
    
    # Step 4: Save model
    print("\n[Train] Saving trained model...")
    model_path = ML_DIR / 'rf_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"  - Saved model to {model_path}")
    
    # Step 5: Save metrics
    metrics_path = ML_DIR / 'training_metrics.json'
    training_info = {
        'model': 'RandomForestClassifier',
        'best_params': best_params,
        'metrics': metrics,
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"  - Saved metrics to {metrics_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model: Random Forest Classifier (Small Dataset Optimized)")
    print(f"Dataset: 205 samples, 34.6% minority class")
    print(f"Best parameters: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"min_samples_split={best_params['min_samples_split']}, "
          f"min_samples_leaf={best_params['min_samples_leaf']}")
    print(f"\nTest Metrics:")
    print(f"  - F1-score (macro): {metrics['test']['f1_macro']:.4f}")
    print(f"  - ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"  - Accuracy: {metrics['test']['accuracy']:.4f}")
    
    # Generalization check
    f1_gap = metrics['train']['f1_macro'] - metrics['test']['f1_macro']
    print(f"\nGeneralization:")
    print(f"  - Train-Test F1 gap: {f1_gap:.4f}")
    if f1_gap <= 0.07:
        print(f"  Good generalization (gap <= 0.07 for small dataset)")
    else:
        print(f"  Overfitting detected (gap > 0.07)")
    
    # Performance target check (F1 > 0.75 for small dataset)
    if metrics['test']['f1_macro'] >= 0.75:
        print("\nModel meets target performance (F1 > 0.75 for 205 samples)")
    else:
        print(f"\nModel below target (F1 = {metrics['test']['f1_macro']:.4f} < 0.75)")
        print("   Consider: More diverse data collection or alternative features")
    
    print("\nNext step: Run predict.py to test predictions")
    print("="*70 + "\n")

if __name__ == '__main__':
    train_model()
