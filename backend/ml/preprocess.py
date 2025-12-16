"""
Data Preprocessing Pipeline for Food Recommendation System
Author: NutriSolve ML Team
Date: December 2025

Research Design: Supervised binary classification for "fit" vs "unfit" foods
Target: Binary label (1=fit, 0=unfit) based on percentile-based nutritional thresholds
Features: Real nutritional data (205 foods) with engineered features

This script:
1. Loads real food nutrition dataset (205 samples)
2. Creates binary dietary flags (gluten-free, nut-free, vegan) for filtering
3. Engineers features for model training (nutrient density, ratios)
4. Creates labels using percentile-based thresholds on raw features
5. Handles missing values via median imputation
6. Encodes categorical features (OneHot)
7. Scales numerical features (StandardScaler)
8. Applies feature selection (SelectKBest chi2: k=8-10)
9. Conditionally applies SMOTE if minority class < 25%
10. Splits data 80/20 stratified
11. Saves preprocessor and processed data
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define paths
BASE_DIR = Path(__file__).parent
ML_DIR = Path(__file__).parent

# Ensure directory exists
ML_DIR.mkdir(exist_ok=True)

# Dataset path
DATASET_PATH = '/home/gohon/Desktop/Nutri-solve (another copy)/nutriflame-ai/backend/ml/Food_Nutrition_Dataset.csv'

def load_and_explore_data():
    """
    Load real food nutrition dataset and display basic information
    
    Dataset: 205 real foods with columns:
    - food_name, category, calories, protein, carbs, fat, iron, vitamin_c
    
    Returns:
        DataFrame: Loaded dataset
    """
    print("[Preprocess] Loading food nutrition dataset...")
    
    df = pd.read_csv(DATASET_PATH)
    
    print(f"\n[Data Exploration]")
    print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\n  - Missing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"    No missing values!")
    else:
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  - Categories: {df['category'].nunique()} unique")
    print(f"  - Sample categories: {df['category'].value_counts().head(5).to_dict()}")
    
    return df


def create_dietary_flags(df):
    """
    Create binary dietary flags based on food category for filtering in prediction phase
    
    These flags are NOT used for labeling, only for user preference filtering
    
    Flags:
    - is_glutenfree: 1 if naturally gluten-free (fruits, vegetables, meats, dairy)
    - is_nutfree: 1 if NOT in nuts/seeds categories
    - is_vegan: 1 if plant-based (fruits, vegetables, grains, legumes)
    
    Returns:
        DataFrame: Dataset with added flags
    """
    print("\n[Preprocess] Creating binary dietary flags...")
    
    # Define category mappings
    gluten_free_categories = [
        'Apples', 'Bananas', 'Berries', 'Citrus fruits', 'Tropical and other fruits',
        'Fruits and Fruit Juices', 'Dried fruits', 'Other fruits and fruit salads',
        'Vegetables', 'Leafy vegetables', 'Tomatoes', 'Root vegetables',
        'Meats', 'Poultry', 'Fish and shellfish', 'Eggs', 'Dairy', 'Milk',
        'Cheese', 'Yogurt', 'Rice', 'Quinoa', 'Legumes', 'Beans'
    ]
    
    nut_categories = [
        'Nuts and seeds', 'Nut and seed products', 'Peanut butter and other nut butters'
    ]
    
    vegan_categories = [
        'Apples', 'Bananas', 'Berries', 'Citrus fruits', 'Tropical and other fruits',
        'Fruits and Fruit Juices', 'Dried fruits', 'Other fruits and fruit salads',
        'Vegetables', 'Leafy vegetables', 'Tomatoes', 'Root vegetables',
        'Grains', 'Rice', 'Quinoa', 'Pasta', 'Bread', 'Cereal',
        'Legumes', 'Beans', 'Lentils', 'Nuts and seeds', 'Plant-based milk'
    ]
    
    # Create flags
    df['is_glutenfree'] = df['category'].apply(
        lambda x: 1 if any(gf in str(x) for gf in gluten_free_categories) else 0
    )
    
    df['is_nutfree'] = df['category'].apply(
        lambda x: 0 if any(nut in str(x) for nut in nut_categories) else 1
    )
    
    df['is_vegan'] = df['category'].apply(
        lambda x: 1 if any(veg in str(x) for veg in vegan_categories) else 0
    )
    
    print(f"  - is_glutenfree: {df['is_glutenfree'].sum()} foods ({df['is_glutenfree'].mean()*100:.1f}%)")
    print(f"  - is_nutfree: {df['is_nutfree'].sum()} foods ({df['is_nutfree'].mean()*100:.1f}%)")
    print(f"  - is_vegan: {df['is_vegan'].sum()} foods ({df['is_vegan'].mean()*100:.1f}%)")
    
    return df


def engineer_features(df):
    """
    Create engineered features for model training
    
    These features help the model learn better patterns but are NOT used for labeling.
    Labels are created from RAW features only.
    
    Engineered Features:
    1. nutrient_density: (protein + iron + vitamin_c) / (calories + 1)
       - Measures nutrient quality per calorie
    
    2. protein_ratio: protein / (calories + 1)
       - High protein relative to calories
    
    3. carb_fat_ratio: carbs / (fat + 0.001)
       - Balance of macronutrients
    
    4. energy_density: calories / 100
       - Caloric concentration
    
    5. micronutrient_score: iron*10 + vitamin_c
       - Combined micronutrient value
    
    Returns:
        DataFrame: Dataset with engineered features
    """
    print("\n[Preprocess] Engineering features for model training...")
    
    # Nutrient density (higher = better)
    df['nutrient_density'] = (df['protein'] + df['iron'] + df['vitamin_c']) / (df['calories'] + 1)
    
    # Protein ratio (higher = more protein per calorie)
    df['protein_ratio'] = df['protein'] / (df['calories'] + 1)
    
    # Carb to fat ratio
    df['carb_fat_ratio'] = df['carbs'] / (df['fat'] + 0.001)
    
    # Energy density (lower = less calorie-dense)
    df['energy_density'] = df['calories'] / 100
    
    # Micronutrient score (higher = better)
    df['micronutrient_score'] = df['iron'] * 10 + df['vitamin_c']
    
    print(f"  - Created 5 engineered features:")
    print(f"    • nutrient_density")
    print(f"    • protein_ratio")
    print(f"    • carb_fat_ratio")
    print(f"    • energy_density")
    print(f"    • micronutrient_score")
    
    return df


def create_percentile_labels(df):
    """
    Create binary 'fit' labels using adaptive percentile-based thresholds on RAW features
    
    Method: Dynamically adjust percentiles until fit class reaches 30-35%
    
    Criteria (food is fit if it meets at least 3 out of 5):
    1. protein ≥ percentile (adjustable)
    2. fat ≤ percentile (adjustable)
    3. calories ≤ percentile (adjustable)
    4. iron ≥ percentile (adjustable)
    5. vitamin_c ≥ percentile (adjustable)
    
    Target: 30-35% fit class through automatic threshold adjustment
    
    Returns:
        DataFrame: Dataset with 'fit' label added
    """
    print("\n[Preprocess] Creating labels using adaptive percentile-based thresholds...")
    print(f"  - Target: 30-35% fit class")
    
    # Initial percentiles (starting point)
    protein_pct = 70
    fat_pct = 40
    calories_pct = 60
    iron_pct = 70
    vitamin_c_pct = 70
    
    print(f"\n  - Initial percentiles:")
    print(f"    • protein ≥ {protein_pct}th percentile")
    print(f"    • fat ≤ {fat_pct}th percentile")
    print(f"    • calories ≤ {calories_pct}th percentile")
    print(f"    • iron ≥ {iron_pct}th percentile")
    print(f"    • vitamin_c ≥ {vitamin_c_pct}th percentile")
    
    # Dynamic adjustment loop
    max_iterations = 20
    iteration = 0
    target_min = 30.0
    target_max = 35.0
    
    print(f"\n  - Starting adaptive threshold adjustment...")
    
    while iteration < max_iterations:
        iteration += 1
        
        # Calculate thresholds from current percentiles
        protein_threshold = df['protein'].quantile(protein_pct / 100.0)
        fat_threshold = df['fat'].quantile(fat_pct / 100.0)
        calories_threshold = df['calories'].quantile(calories_pct / 100.0)
        iron_threshold = df['iron'].quantile(iron_pct / 100.0)
        vitamin_c_threshold = df['vitamin_c'].quantile(vitamin_c_pct / 100.0)
        
        # Create individual criteria
        criteria = pd.DataFrame({
            'high_protein': (df['protein'] >= protein_threshold).astype(int),
            'low_fat': (df['fat'] <= fat_threshold).astype(int),
            'low_calories': (df['calories'] <= calories_threshold).astype(int),
            'high_iron': (df['iron'] >= iron_threshold).astype(int),
            'high_vitamin_c': (df['vitamin_c'] >= vitamin_c_threshold).astype(int)
        })
        
        # Label as 'fit' if at least 3 out of 5 criteria met
        criteria_met = criteria.sum(axis=1)
        temp_fit = (criteria_met >= 3).astype(int)
        
        # Calculate current fit percentage
        fit_count = temp_fit.sum()
        fit_pct = fit_count / len(df) * 100
        
        print(f"    Iteration {iteration}: fit={fit_pct:.1f}% ({fit_count} foods) | "
              f"p={protein_pct}, f={fat_pct}, c={calories_pct}, i={iron_pct}, v={vitamin_c_pct}")
        
        # Check if target reached
        if target_min <= fit_pct <= target_max:
            print(f"\n  Target reached: {fit_pct:.1f}% fit (within {target_min}-{target_max}%)")
            df['fit'] = temp_fit
            break
        
        # Adjust percentiles
        if fit_pct < target_min:
            # Too few fit foods - make criteria easier
            protein_pct = max(30, protein_pct - 5)
            iron_pct = max(30, iron_pct - 5)
            vitamin_c_pct = max(30, vitamin_c_pct - 5)
            fat_pct = min(70, fat_pct + 5)
            calories_pct = min(80, calories_pct + 5)
        elif fit_pct > target_max:
            # Too many fit foods - make criteria stricter
            protein_pct = min(90, protein_pct + 5)
            iron_pct = min(90, iron_pct + 5)
            vitamin_c_pct = min(90, vitamin_c_pct + 5)
            fat_pct = max(20, fat_pct - 5)
            calories_pct = max(30, calories_pct - 5)
        
        # Last iteration check
        if iteration == max_iterations:
            print(f"\n  Max iterations reached. Using last result: {fit_pct:.1f}% fit")
            df['fit'] = temp_fit
    
    # Final statistics
    fit_count = df['fit'].sum()
    unfit_count = len(df) - fit_count
    fit_pct = fit_count / len(df) * 100
    unfit_pct = unfit_count / len(df) * 100
    
    print(f"\n  - Final adjusted percentiles:")
    print(f"    • protein ≥ {protein_pct}th percentile ({protein_threshold:.2f})")
    print(f"    • fat ≤ {fat_pct}th percentile ({fat_threshold:.2f})")
    print(f"    • calories ≤ {calories_pct}th percentile ({calories_threshold:.2f})")
    print(f"    • iron ≥ {iron_pct}th percentile ({iron_threshold:.2f})")
    print(f"    • vitamin_c ≥ {vitamin_c_pct}th percentile ({vitamin_c_threshold:.2f})")
    
    print(f"\n  - Final label distribution:")
    print(f"    • fit=1: {fit_count} ({fit_pct:.1f}%)")
    print(f"    • fit=0: {unfit_count} ({unfit_pct:.1f}%)")
    
    if target_min <= fit_pct <= target_max:
        print(f"    Successfully achieved target range ({target_min}-{target_max}%)")
    
    return df

def preprocess_data():
    """
    Main preprocessing pipeline for food recommendation system
    
    Pipeline Steps:
    1. Load real food nutrition dataset (205 samples)
    2. Create binary dietary flags (gluten-free, nut-free, vegan)
    3. Engineer features for model training
    4. Create labels using percentile-based thresholds
    5. Handle missing values (median imputation)
    6. Define feature lists (numerical, categorical, binary)
    7. Train-test split (80/20 stratified)
    8. Create preprocessing pipeline (ColumnTransformer)
    9. Feature selection (SelectKBest chi2, k=8-10)
    10. Conditional SMOTE (only if minority < 25%)
    11. Save all artifacts
    
    Outputs:
    - processed_data.csv (full labeled dataset)
    - train_data.csv, test_data.csv (with selected features)
    - preprocessor.pkl (fitted ColumnTransformer)
    - feature_selector.pkl (fitted SelectKBest)
    - feature_info.json (feature names, scores)
    """
    print("\n" + "="*70)
    print("FOOD RECOMMENDATION ML PIPELINE - DATA PREPROCESSING")
    print("="*70 + "\n")
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Create binary dietary flags
    df = create_dietary_flags(df)
    
    # Step 3: Engineer features
    df = engineer_features(df)
    
    # Step 4: Create labels using percentile-based thresholds
    df = create_percentile_labels(df)
    
    # Step 5: Define feature columns
    numerical_features = [
        'calories', 'protein', 'carbs', 'fat', 'iron', 'vitamin_c',
        'nutrient_density', 'protein_ratio', 'carb_fat_ratio', 
        'energy_density', 'micronutrient_score'
    ]
    
    categorical_features = ['category']
    
    binary_features = ['is_glutenfree', 'is_nutfree', 'is_vegan']
    
    print(f"\n[Feature Definition]")
    print(f"  - Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"  - Binary features ({len(binary_features)}): {binary_features}")
    
    # Step 6: Handle missing values
    print("\n[Preprocess] Handling missing values...")
    missing_found = False
    for col in numerical_features:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  - Imputed {col} with median {median_val:.2f}")
            missing_found = True
    
    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('unknown')
            print(f"  - Filled {col} with 'unknown'")
            missing_found = True
    
    if not missing_found:
        print(f"  No missing values found in feature columns")
    
    # Step 7: Prepare features and target
    X = df[numerical_features + categorical_features + binary_features]
    y = df['fit']
    
    print(f"\n[Feature Matrix]")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Total samples: {X.shape[0]}")
    print(f"  - Target distribution: fit=1 ({y.sum()}, {y.mean()*100:.1f}%), fit=0 ({(1-y).sum()}, {(1-y).mean()*100:.1f}%)")
    
    # Step 8: Train-test split (80/20 stratified)
    print(f"\n[Preprocess] Splitting data 80/20 stratified (random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  - Train: {len(X_train)} samples (fit=1: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
    print(f"  - Test: {len(X_test)} samples (fit=1: {y_test.sum()}, {y_test.mean()*100:.1f}%)")
    
    # Step 9: Create preprocessing pipeline (ColumnTransformer)
    print("\n[Preprocess] Creating preprocessing pipeline...")
    
    # ColumnTransformer: Apply different transformations to different feature types
    # - Numerical: StandardScaler (mean=0, std=1)
    # - Categorical: OneHotEncoder (drop='first' to avoid multicollinearity)
    # - Binary: passthrough (already 0/1)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    
    # Fit on training data only (prevent data leakage)
    print(f"  - Fitting preprocessor on training data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    num_names = numerical_features
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    bin_names = binary_features
    feature_names = num_names + cat_names + bin_names
    
    print(f"  - Features after transformation: {len(feature_names)}")
    print(f"  - Numerical: {len(num_names)}, Categorical (encoded): {len(cat_names)}, Binary: {len(bin_names)}")
    
    # Step 10: Feature selection (SelectKBest with chi2)
    # For 205 samples, select k=8-10 features to prevent overfitting
    # chi2 measures dependency between feature and target
    print("\n[Preprocess] Applying feature selection (SelectKBest chi2)...")
    
    # Determine k based on dataset size (205 samples → k=9)
    n_features_to_select = min(9, len(feature_names))
    print(f"  - Selecting k={n_features_to_select} features (optimal for {len(X_train)} samples)")
    
    # Convert to dense array if sparse and make non-negative for chi2
    if hasattr(X_train_transformed, 'toarray'):
        X_train_dense = X_train_transformed.toarray()
        X_test_dense = X_test_transformed.toarray()
    else:
        X_train_dense = X_train_transformed
        X_test_dense = X_test_transformed
    
    # Make non-negative (chi2 requirement)
    X_train_nonneg = X_train_dense - X_train_dense.min(axis=0) + 1e-9
    X_test_nonneg = X_test_dense - X_test_dense.min(axis=0) + 1e-9
    
    selector = SelectKBest(score_func=chi2, k=n_features_to_select)
    X_train_selected = selector.fit_transform(X_train_nonneg, y_train)
    X_test_selected = selector.transform(X_test_nonneg)
    
    # Get selected feature names and scores
    selected_idx = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_idx]
    chi2_scores = selector.scores_[selected_idx]
    
    print(f"\n  - Top {len(selected_features)} selected features:")
    for i, (feat, score) in enumerate(sorted(zip(selected_features, chi2_scores), 
                                              key=lambda x: x[1], reverse=True), 1):
        print(f"    {i}. {feat}: chi2={score:.2f}")
    
    # Step 11: Conditional SMOTE oversampling (training set only)
    # Only apply if minority class < 25%, target ~30-40% minority
    print("\n[SMOTE Strategy]")
    
    minority_class = min(y_train.sum(), len(y_train) - y_train.sum())
    minority_pct = minority_class / len(y_train) * 100
    
    print(f"  - Training set: fit=1 ({y_train.sum()}), fit=0 ({len(y_train) - y_train.sum()})")
    print(f"  - Minority class: {minority_pct:.1f}%")
    
    if minority_pct < 25:
        print(f"  - Minority < 25%, applying SMOTE to balance to ~30-40%")
        
        # Calculate sampling strategy to reach 30% minority
        majority_class = max(y_train.sum(), len(y_train) - y_train.sum())
        target_minority = int(majority_class * 0.42)  # Target 30% minority (42% of majority)
        
        # Determine k_neighbors (conservative for small dataset)
        k_neighbors = min(5, minority_class - 1) if minority_class > 1 else 1
        
        print(f"  - Target minority samples: {target_minority}")
        print(f"  - Using k_neighbors={k_neighbors} (conservative for {len(X_train)} samples)")
        
        smote = SMOTE(
            sampling_strategy=target_minority / majority_class,
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
        
        new_minority_pct = min(y_train_resampled.sum(), len(y_train_resampled) - y_train_resampled.sum()) / len(y_train_resampled) * 100
        print(f"  - After SMOTE: fit=1 ({y_train_resampled.sum()}), fit=0 ({len(y_train_resampled) - y_train_resampled.sum()})")
        print(f"  - New minority: {new_minority_pct:.1f}%")
        print(f"  - Training samples: {len(y_train)} → {len(y_train_resampled)}")
    else:
        print(f"  - Minority ≥ 25%, skipping SMOTE (natural distribution is acceptable)")
        X_train_resampled = X_train_selected
        y_train_resampled = y_train
    
    # Step 12: Save all artifacts
    print("\n[Saving Artifacts]")
    print("="*70)
    
    # Save full processed dataset with labels
    df.to_csv(ML_DIR / 'processed_data.csv', index=False)
    print(f"  Saved processed_data.csv ({len(df)} rows)")
    
    # Save train/test splits (with selected features)
    train_df = pd.DataFrame(X_train_resampled, columns=selected_features)
    # Handle both numpy array (from SMOTE) and pandas Series (no SMOTE)
    if isinstance(y_train_resampled, np.ndarray):
        train_df['fit'] = y_train_resampled
    else:
        train_df['fit'] = y_train_resampled.values
    train_df.to_csv(ML_DIR / 'train_data.csv', index=False)
    print(f"  Saved train_data.csv ({len(train_df)} rows)")
    
    test_df = pd.DataFrame(X_test_selected, columns=selected_features)
    test_df['fit'] = y_test.values
    test_df.to_csv(ML_DIR / 'test_data.csv', index=False)
    print(f"  Saved test_data.csv ({len(test_df)} rows)")
    
    # Save preprocessing pipeline
    joblib.dump(preprocessor, ML_DIR / 'preprocessor.pkl')
    print(f"  Saved preprocessor.pkl")
    
    # Save feature selector
    joblib.dump(selector, ML_DIR / 'feature_selector.pkl')
    print(f"  Saved feature_selector.pkl")
    
    # Save feature information
    feature_info = {
        'all_features': feature_names,
        'selected_features': selected_features,
        'chi2_scores': chi2_scores.tolist(),
        'feature_counts': {
            'numerical': len(num_names),
            'categorical_encoded': len(cat_names),
            'binary': len(bin_names),
            'total_before_selection': len(feature_names),
            'total_after_selection': len(selected_features)
        }
    }
    with open(ML_DIR / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  Saved feature_info.json")
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Training samples: {len(train_df)} (after preprocessing)")
    print(f"  • Test samples: {len(test_df)}")
    print(f"  • Selected features: {len(selected_features)} (from {len(feature_names)} total)")
    print(f"\nLabel Distribution:")
    print(f"  • Overall: fit=1 ({df['fit'].sum()}), fit=0 ({len(df)-df['fit'].sum()})")
    print(f"  • Training: fit=1 ({train_df['fit'].sum()}), fit=0 ({len(train_df)-train_df['fit'].sum()})")
    print(f"  • Test: fit=1 ({test_df['fit'].sum()}), fit=0 ({len(test_df)-test_df['fit'].sum()})")
    print(f"\nSaved Artifacts:")
    print(f"  • processed_data.csv - Full labeled dataset")
    print(f"  • train_data.csv - Training set with selected features")
    print(f"  • test_data.csv - Test set with selected features")
    print(f"  • preprocessor.pkl - Fitted ColumnTransformer")
    print(f"  • feature_selector.pkl - Fitted SelectKBest")
    print(f"  • feature_info.json - Feature metadata")
    print(f"\nNext Step: Run train.py for model training")
    print("="*70 + "\n")

if __name__ == '__main__':
    preprocess_data()
