"""
Real-time Prediction Service for Food Recommendations
Author: NutriSolve ML Team
Date: December 2025

Updated to work with new preprocessing pipeline (205 real foods, adaptive thresholds)
All preprocessing is done offline - this script only loads and predicts.

Integration:
Called via child_process.spawn() from /backend/controllers/recController.ts
Communicates via stdin/stdout JSON protocol
"""

import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths (must match training pipeline)
BASE_DIR = Path(__file__).parent
ML_DIR = Path(__file__).parent


def load_model_and_features():
    """
    Load trained model and selected feature names
    
    Returns: model, selected_features list
    
    Note: NO preprocessor or feature_selector needed
    processed_data.csv is already fully preprocessed
    """
    try:
        # Load trained Random Forest model
        model = joblib.load(ML_DIR / 'rf_model.pkl')
        
        # Load selected features from feature_info.json
        with open(ML_DIR / 'feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        selected_features = feature_info['selected_features']
        
        # Note: selected_features may include category_* features
        # We'll need to encode the 'category' column later
        
        return model, selected_features
        
    except FileNotFoundError as e:
        print(json.dumps({'error': f'Model files not found. Run train.py first. {str(e)}'}), 
              file=sys.stderr)
        sys.exit(1)


def load_food_database():
    """
    Load preprocessed food database
    
    Returns: DataFrame with 205 foods, all features already computed
    
    Note: processed_data.csv contains:
    - Raw features: calories, protein, carbs, fat, iron, vitamin_c
    - Engineered features: nutrient_density, protein_ratio, carb_fat_ratio, 
                           energy_density, micronutrient_score
    - Binary flags: is_glutenfree, is_nutfree, is_vegan
    - Labels: fit (not used for prediction)
    """
    try:
        df = pd.read_csv(ML_DIR / 'processed_data.csv')
        return df
    except FileNotFoundError as e:
        print(json.dumps({'error': f'Food database not found: {str(e)}'}), 
              file=sys.stderr)
        sys.exit(1)


def estimate_cost(df):
    """
    Estimate cost per serving based on category and calories
    
    Heuristic approach (since cost not in dataset):
    - Base costs by category
    - Scale by calorie density
    
    Returns: DataFrame with 'estimated_cost' column added
    """
    # Base costs per serving by category (approximate USD)
    base_costs = {
        'Fruits and Fruit Juices': 0.30,
        'Vegetables and Vegetable Products': 0.25,
        'Dairy and Egg Products': 0.50,
        'Legumes and Legume Products': 0.35,
        'Apples': 0.30,
        'Bananas': 0.20,
        'Berries': 0.60,
        'Beverages': 0.40,
        'Snacks': 0.80,
        'Cakes and pies': 1.00,
        'Fast Foods': 1.20
    }
    
    # Default cost for unknown categories
    default_cost = 0.50
    
    # Estimate cost: base_cost * (calories / 150)
    # Scales by energy density (150 cal = baseline)
    df['estimated_cost'] = df.apply(
        lambda row: base_costs.get(row.get('category', 'unknown'), default_cost) 
                    * (row['calories'] / 150.0),
        axis=1
    )
    
    # Clip to reasonable range [0.10, 3.00]
    df['estimated_cost'] = df['estimated_cost'].clip(0.10, 3.00)
    
    return df

def filter_by_user_constraints(df, user_profile):
    """
    Filter foods based on user's dietary restrictions and budget
    
    User constraints from onboarding:
    - dietaryRestrictions: ['Vegan', 'Gluten Free', 'Nut Allergy']
    - weeklyBudget: max cost per serving
    
    Returns: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Extract user constraints
    restrictions = user_profile.get('dietaryRestrictions', [])
    budget = user_profile.get('weeklyBudget', 100)  # Default $100/week
    max_cost_per_serving = budget / 21  # Assume 3 meals/day × 7 days
    
    # Apply dietary restrictions
    if 'Vegan' in restrictions or 'vegan' in restrictions:
        if 'is_vegan' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_vegan'] == 1]
            print(f"Applied Vegan filter: {len(filtered_df)} foods remaining", file=sys.stderr)
    
    if 'Gluten Free' in restrictions or 'gluten-free' in restrictions:
        if 'is_glutenfree' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_glutenfree'] == 1]
            print(f"Applied Gluten-free filter: {len(filtered_df)} foods remaining", file=sys.stderr)
    
    if 'Nut Allergy' in restrictions or 'nut-free' in restrictions:
        if 'is_nutfree' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_nutfree'] == 1]
            print(f"Applied Nut-free filter: {len(filtered_df)} foods remaining", file=sys.stderr)
    
    # Apply budget constraint
    if 'estimated_cost' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['estimated_cost'] <= max_cost_per_serving]
        print(f"Applied budget filter (≤${max_cost_per_serving:.2f}): {len(filtered_df)} foods remaining", 
              file=sys.stderr)
    
    return filtered_df

def apply_goal_adjustments(probs, df, user_profile):
    """
    Apply goal-based adjustments to model probabilities
    
    Goal-specific boosts:
    - Weight Loss: Low calories, high protein ratio
    - Muscle Gain: High protein, high nutrient density
    - Heart Health: Low calories, high micronutrient score
    - General Health: Small smoothing only
    
    Returns: Adjusted probabilities
    """
    goal = user_profile.get('primaryGoal', 'General Health')
    adjusted_probs = probs.copy()
    
    if goal == 'Weight Loss':
        # Boost low-calorie, high-protein-ratio foods
        weight_loss_boost = (
            (df['calories'] < 200) * 0.15 +  # Low calorie bonus
            (df['protein_ratio'] > 0.05) * 0.10  # High protein ratio bonus
        )
        adjusted_probs = adjusted_probs * (1 + weight_loss_boost)
        print(f"Applied Weight Loss adjustments", file=sys.stderr)
    
    elif goal == 'Muscle Gain':
        # Boost high-protein, nutrient-dense foods
        muscle_boost = (
            (df['protein'] > 10) * 0.20 +  # High protein bonus
            (df['nutrient_density'] > 0.1) * 0.10  # Nutrient density bonus
        )
        adjusted_probs = adjusted_probs * (1 + muscle_boost)
        print(f"Applied Muscle Gain adjustments", file=sys.stderr)
    
    elif goal == 'Heart Health':
        # Boost foods with high micronutrients, low energy density
        heart_boost = (
            (df['micronutrient_score'] > 20) * 0.15 +  # High micronutrients bonus
            (df['energy_density'] < 2) * 0.10  # Low energy density bonus
        )
        adjusted_probs = adjusted_probs * (1 + heart_boost)
        print(f"Applied Heart Health adjustments", file=sys.stderr)
    
    else:  # General Health or unknown
        # Small smoothing only (5% boost across board)
        adjusted_probs = adjusted_probs * 1.05
        print(f"Applied General Health smoothing", file=sys.stderr)
    
    # Clip to valid probability range
    adjusted_probs = np.clip(adjusted_probs, 0, 1)
    
    return adjusted_probs


def apply_cost_penalty(adjusted_probs, df):
    """
    Apply cost penalty to final scores
    
    Strategy:
    - Normalize costs relative to median
    - Apply gentle 15% penalty for expensive foods
    - Ensure nutrition still dominates over cost
    
    Returns: Final scores with cost penalty applied
    """
    costs = df['estimated_cost'].values
    median_cost = np.median(costs)
    
    # Normalize cost: (cost - median) / median
    # Results in range roughly [-1, 1] for most foods
    norm_cost = (costs - median_cost) / (median_cost + 0.01)
    norm_cost = np.clip(norm_cost, 0, 1)  # Only penalize above-median costs
    
    # Apply gentle penalty: final_score = adjusted_prob - (norm_cost * 0.15)
    # 15% penalty ensures cost influences ranking but doesn't override nutrition
    final_scores = adjusted_probs - (norm_cost * 0.15)
    final_scores = np.clip(final_scores, 0, 1)
    
    print(f"Applied cost penalty (median: ${median_cost:.2f})", file=sys.stderr)
    
    return final_scores


def generate_reasons(food):
    """
    Generate human-readable reasons for recommendation
    
    Based on available features in processed_data.csv
    """
    reasons = []
    
    # High protein
    if food.get('protein', 0) > 5:
        reasons.append(f"High protein ({food['protein']:.1f}g)")
    
    # Nutrient density
    if food.get('nutrient_density', 0) > 0.1:
        reasons.append(f"Nutrient-dense (score: {food['nutrient_density']:.2f})")
    
    # Vitamin C
    if food.get('vitamin_c', 0) > 20:
        reasons.append(f"Rich in vitamin C ({food['vitamin_c']:.0f}mg)")
    
    # Low calorie
    if food.get('calories', 0) < 150:
        reasons.append(f"Low calorie ({food['calories']:.0f} kcal)")
    
    # Iron
    if food.get('iron', 0) > 1.0:
        reasons.append(f"Good source of iron ({food['iron']:.1f}mg)")
    
    # Budget-friendly
    if food.get('estimated_cost', 999) < 0.50:
        reasons.append(f"Budget-friendly (${food['estimated_cost']:.2f})")
    
    # Micronutrient score
    if food.get('micronutrient_score', 0) > 30:
        reasons.append(f"Rich in micronutrients (score: {food['micronutrient_score']:.0f})")
    
    return reasons

def predict_top_meals(user_input, top_k=5):
    """
    Main prediction pipeline
    
    Steps:
    1. Load model and selected features (no preprocessing artifacts)
    2. Load preprocessed food database (already has all features)
    3. Estimate costs for all foods
    4. Filter foods by user constraints (dietary + budget)
    5. Extract features using selected_features (no transformation needed)
    6. Predict probabilities using trained model
    7. Apply goal-based adjustments
    8. Apply cost penalty
    9. Rank by final score and return top-k
    
    Args:
        user_input: Dict with 'userProfile' and 'query'
        top_k: Number of recommendations to return
    
    Returns:
        Dict with ranked meals, scores, and metadata
    """
    # Step 1: Load model and features
    model, selected_features = load_model_and_features()
    print(f"Loaded model with {len(selected_features)} selected features", file=sys.stderr)
    
    # Step 2: Load preprocessed food database
    food_db = load_food_database()
    print(f"Loaded {len(food_db)} foods from database", file=sys.stderr)
    
    # Step 3: Estimate costs
    food_db = estimate_cost(food_db)
    
    # Step 4: Extract user data and filter
    user_profile = user_input.get('userProfile', {})
    query = user_input.get('query', '')
    
    eligible_foods = filter_by_user_constraints(food_db, user_profile)
    
    if len(eligible_foods) == 0:
        return {
            'recommendations': [],
            'message': 'No foods match your dietary restrictions and budget. Try relaxing some constraints.',
            'total_eligible': 0
        }
    
    print(f"After filtering: {len(eligible_foods)} eligible foods", file=sys.stderr)
    
    # Step 5: Extract features (handle category encoding if needed)
    # Separate numerical/engineered features from category features
    numerical_features = [f for f in selected_features if not f.startswith('category_')]
    category_features = [f for f in selected_features if f.startswith('category_')]
    
    # Build feature matrix
    X_numerical = eligible_foods[numerical_features].values
    
    # Handle category encoding if model requires category features
    if category_features:
        # One-hot encode the category column
        from sklearn.preprocessing import LabelBinarizer
        
        # Get unique categories from selected features
        categories_needed = [f.replace('category_', '') for f in category_features]
        
        # Create binary columns for each category
        X_category = np.zeros((len(eligible_foods), len(category_features)))
        for idx, row in eligible_foods.iterrows():
            cat = row.get('category', '')
            for i, cat_name in enumerate(categories_needed):
                if cat_name in cat:
                    X_category[idx, i] = 1
        
        # Combine numerical and categorical features
        X = np.hstack([X_numerical, X_category])
    else:
        X = X_numerical
    
    print(f"Feature matrix shape: {X.shape} (numerical: {len(numerical_features)}, categorical: {len(category_features)})", file=sys.stderr)
    
    # Step 6: Predict probabilities (model trained on preprocessed data)
    model_probs = model.predict_proba(X)[:, 1]  # Probability of fit=1
    print(f"Model predictions - mean: {model_probs.mean():.3f}, std: {model_probs.std():.3f}", 
          file=sys.stderr)
    
    # Step 7: Apply goal-based adjustments
    goal_adjusted_probs = apply_goal_adjustments(model_probs, eligible_foods, user_profile)
    
    # Step 8: Apply cost penalty to final ranking
    final_scores = apply_cost_penalty(goal_adjusted_probs, eligible_foods)
    
    # Step 9: Rank by final score (descending)
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    # Build recommendations
    recommendations = []
    for idx in top_indices:
        food = eligible_foods.iloc[idx]
        model_prob = model_probs[idx]
        adjusted_score = goal_adjusted_probs[idx]
        final_score = final_scores[idx]
        
        # Generate reasons
        reasons = generate_reasons(food)
        
        # Dietary flags
        dietary = []
        if food.get('is_vegan', 0) == 1:
            dietary.append('Vegan')
        if food.get('is_glutenfree', 0) == 1:
            dietary.append('Gluten-free')
        if food.get('is_nutfree', 0) == 1:
            dietary.append('Nut-free')
        
        # Confidence level
        if final_score > 0.8:
            confidence = 'high'
        elif final_score > 0.6:
            confidence = 'medium'
        else:
            confidence = 'moderate'
        
        recommendation = {
            'name': food.get('food_name', 'Unknown'),
            'category': food.get('category', 'Unknown'),
            'model_probability': float(model_prob),
            'goal_adjusted_score': float(adjusted_score),
            'final_score': float(final_score),
            'confidence': confidence,
            'nutrition': {
                'calories': float(food.get('calories', 0)),
                'protein': float(food.get('protein', 0)),
                'carbs': float(food.get('carbs', 0)),
                'fat': float(food.get('fat', 0)),
                'iron': float(food.get('iron', 0)),
                'vitamin_c': float(food.get('vitamin_c', 0)),
                'nutrient_density': float(food.get('nutrient_density', 0)),
                'micronutrient_score': float(food.get('micronutrient_score', 0))
            },
            'estimated_cost': float(food.get('estimated_cost', 0)),
            'reasons': reasons,
            'dietary_info': dietary
        }
        recommendations.append(recommendation)
    
    return {
        'recommendations': recommendations,
        'query': query,
        'total_foods_in_db': len(food_db),
        'total_eligible': len(eligible_foods),
        'model_version': '2.0-adaptive-thresholds',
        'user_goal': user_profile.get('primaryGoal', 'General Health'),
        'features_used': len(selected_features)
    }

def main():
    """
    Main entry point for command-line prediction
    
    Usage:
    1. From TypeScript: echo '{"userProfile": {...}, "query": "..."}' | python predict.py
    2. From command line: python predict.py < input.json
    
    Input JSON format:
    {
      "userProfile": {
        "age": 30,
        "gender": "Female",
        "primaryGoal": "Weight Loss",
        "dietaryRestrictions": ["Vegan"],
        "weeklyBudget": 75
      },
      "query": "healthy breakfast options",
      "top_k": 5
    }
    
    Output JSON format:
    {
      "recommendations": [
        {
          "name": "Food name",
          "model_probability": 0.85,
          "goal_adjusted_score": 0.90,
          "final_score": 0.87,
          "confidence": "high",
          "nutrition": {...},
          "estimated_cost": 0.45,
          "reasons": [...],
          "dietary_info": [...]
        }
      ],
      "total_eligible": 67,
      "model_version": "2.0-adaptive-thresholds"
    }
    """
    try:
        # Read input from stdin
        if sys.stdin.isatty():
            # Interactive mode: use default test input
            print("No input provided. Using test data...", file=sys.stderr)
            user_input = {
                'userProfile': {
                    'age': 30,
                    'gender': 'Female',
                    'primaryGoal': 'Weight Loss',
                    'dietaryRestrictions': [],
                    'weeklyBudget': 75
                },
                'query': 'healthy breakfast',
                'top_k': 5
            }
        else:
            # Read from stdin (TypeScript integration)
            input_json = sys.stdin.read()
            user_input = json.loads(input_json)
        
        # Get top_k parameter
        top_k = user_input.get('top_k', 5)
        
        # Predict
        result = predict_top_meals(user_input, top_k=top_k)
        
        # Output JSON to stdout
        print(json.dumps(result, indent=2))
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        error_response = {'error': f'Invalid JSON input: {str(e)}'}
        print(json.dumps(error_response), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_response = {'error': f'Prediction failed: {str(e)}'}
        print(json.dumps(error_response), file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
