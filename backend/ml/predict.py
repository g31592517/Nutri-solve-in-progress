"""
Real-time Prediction Service for Meal Recommendations
Author: NutriSolve ML Team
Date: October 2025

This script provides prediction endpoint for integration with TypeScript backend
Input: JSON with user profile and query parameters
Output: JSON with ranked meal recommendations and probabilities

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

# Define paths
BASE_DIR = Path(__file__).parent.parent
ML_DIR = BASE_DIR / 'ml'
DATA_DIR = BASE_DIR / 'data'

def load_models():
    """
    Load trained model, preprocessor, and feature selector
    Returns: model, preprocessor, feature_selector, feature_names
    """
    try:
        model = joblib.load(ML_DIR / 'rf_model.pkl')
        preprocessor = joblib.load(ML_DIR / 'preprocessor.pkl')
        feature_selector = joblib.load(ML_DIR / 'feature_selector.pkl')
        
        with open(ML_DIR / 'feature_names.json', 'r') as f:
            feature_info = json.load(f)
        
        return model, preprocessor, feature_selector, feature_info
    except FileNotFoundError as e:
        print(json.dumps({'error': f'Model files not found. Run train.py first. {str(e)}'}), file=sys.stderr)
        sys.exit(1)

def load_food_database():
    """
    Load complete food database for ranking
    Returns: DataFrame with all food items and their features
    """
    try:
        # Load processed data (includes all foods with computed features)
    
    # Prepare features for prediction
    numerical_features = [
        'calories', 'protein_g', 'fat_g', 'carbs_g', 'fiber_g', 'sugars_g',
        'sodium_mg', 'vitamin_a_iu', 'vitamin_c_mg', 'calcium_mg', 'iron_mg',
        'potassium_mg', 'magnesium_mg', 'zinc_mg', 'phosphorus_mg',
        'cost_per_serving', 'nutrient_density', 'sugar_to_carb_ratio'
    ]
    categorical_features = ['food_category']
    binary_features = ['is_glutenfree', 'is_nutfree', 'is_vegan']
    
        df = pd.read_csv(ML_DIR / 'processed_data.csv')
        return df
    except FileNotFoundError:
        # Fallback to raw USDA data
        try:
            df = pd.read_csv(DATA_DIR / 'usda-foods.csv')
            return df
        except FileNotFoundError as e:
            print(json.dumps({'error': f'Food database not found: {str(e)}'}), file=sys.stderr)
            sys.exit(1)

def filter_by_user_constraints(df, user_profile):
    """
    Filter foods based on user's dietary restrictions and preferences
    
    User constraints from onboarding:
    - dietaryRestrictions: ['vegan', 'gluten-free', 'nut-free', etc.]
    - weeklyBudget: max cost per serving
    - favoriteCuisines: ['Italian', 'Asian', etc.] (optional filtering)
    
    Returns: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Extract user constraints
    restrictions = user_profile.get('dietaryRestrictions', [])
    budget = user_profile.get('weeklyBudget', 100)  # Default $100/week
    max_cost_per_serving = budget / 21  # Assume 3 meals/day * 7 days
    
    # Filter by dietary restrictions
    if 'Vegan' in restrictions or 'vegan' in restrictions:
        if 'is_vegan' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_vegan'] == 1]
    
    if 'Gluten Free' in restrictions or 'gluten-free' in restrictions:
        if 'is_glutenfree' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_glutenfree'] == 1]
    
    if 'Nut Allergy' in restrictions or 'nut-free' in restrictions:
        if 'is_nutfree' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_nutfree'] == 1]
    
    # Filter by budget
    if 'cost_per_serving' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['cost_per_serving'] <= max_cost_per_serving]
    
    return filtered_df

def adjust_for_goals(probs, df, user_profile):
    """
    Adjust prediction probabilities based on user's primary goal
    
    Goal-based adjustments:
    - Weight Loss: Boost low-calorie, high-protein foods
    - Muscle Gain: Boost high-protein, high-calorie foods
    - Heart Health: Boost low-sodium, high-fiber foods
    - Budget: Already filtered by cost, boost nutrient density
    
    Returns: Adjusted probabilities
    """
    goal = user_profile.get('primaryGoal', '')
    adjusted_probs = probs.copy()
    
    if goal == 'Weight Loss':
        # Boost foods with calories < 300 and protein > 15g
        low_cal_mask = (df['calories'] < 300) & (df['protein_g'] > 15)
        adjusted_probs[low_cal_mask] *= 1.2
    
    elif goal == 'Muscle Gain':
        # Boost high-protein foods (>20g protein)
        high_protein_mask = df['protein_g'] > 20
        adjusted_probs[high_protein_mask] *= 1.3
    
    elif goal == 'Heart Health':
        # Boost low-sodium, high-fiber foods
        heart_healthy_mask = (df['sodium_mg'] < 500) & (df['fiber_g'] > 5)
        adjusted_probs[heart_healthy_mask] *= 1.2
    
    # Normalize probabilities to [0, 1]
    adjusted_probs = np.clip(adjusted_probs, 0, 1)
    
    return adjusted_probs

def predict_top_meals(user_input, top_k=5):
    """
    Main prediction pipeline
    
    Steps:
    1. Load trained model and food database
    2. Filter foods by user constraints (allergies, budget)
    3. Prepare features for prediction
    4. Predict fit probabilities for all eligible foods
    5. Adjust probabilities based on user goals
    6. Rank and return top-k recommendations
    
    Args:
        user_input: Dict with 'userProfile' (from onboarding) and 'query' (text)
        top_k: Number of recommendations to return
    
    Returns:
        Dict with ranked meals, probabilities, and reasons
    """
    # Load artifacts
    model, preprocessor, feature_selector, feature_info = load_models()
    food_db = load_food_database()
    
    # Extract user data
    user_profile = user_input.get('userProfile', {})
    query = user_input.get('query', '')
    
    # Filter foods by user constraints
    eligible_foods = filter_by_user_constraints(food_db, user_profile)
    
    if len(eligible_foods) == 0:
        return {
            'recommendations': [],
            'message': 'No foods match your dietary restrictions and budget. Try relaxing some constraints.'
        }
    
    # Prepare features for prediction
    numerical_features = [
        'calories', 'protein_g', 'fat_g', 'carbs_g', 'fiber_g', 'sugars_g',
        'sodium_mg', 'vitamin_a_iu', 'vitamin_c_mg', 'calcium_mg', 'iron_mg',
        'potassium_mg', 'magnesium_mg', 'zinc_mg', 'phosphorus_mg',
        'cost_per_serving', 'nutrient_density', 'sugar_to_carb_ratio'
    ]
    categorical_features = ['food_category']
    binary_features = ['is_glutenfree', 'is_nutfree', 'is_vegan']
    
    # Ensure all required columns exist (fill missing with defaults)
    for col in numerical_features:
        if col not in eligible_foods.columns:
            if col == 'nutrient_density':
                eligible_foods[col] = (eligible_foods.get('protein_g', 0) + eligible_foods.get('fiber_g', 0)) / (eligible_foods.get('calories', 1) + 1)
            elif col == 'sugar_to_carb_ratio':
                eligible_foods[col] = eligible_foods.get('sugars_g', 0) / (eligible_foods.get('carbs_g', 1) + 1)
            else:
                eligible_foods[col] = 0
    
    for col in categorical_features:
        if col not in eligible_foods.columns:
            eligible_foods[col] = 'unknown'
    
    for col in binary_features:
        if col not in eligible_foods.columns:
            eligible_foods[col] = 0
    
    # Select feature columns
    X = eligible_foods[numerical_features + categorical_features + binary_features]
    
    # Apply preprocessing pipeline
    X_transformed = preprocessor.transform(X)
    
    # Handle non-negative requirement for chi2
    X_nonneg = X_transformed - X_transformed.min() + 1e-9
    
    # Apply feature selection
    X_selected = feature_selector.transform(X_nonneg)
    
    # Predict probabilities
    probs = model.predict_proba(X_selected)[:, 1]  # Probability of fit=1
    
    # Adjust for user goals
    probs = adjust_for_goals(probs, eligible_foods, user_profile)
    
    # Rank by probability (descending)
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    # Build recommendations
    recommendations = []
    for idx in top_indices:
        food = eligible_foods.iloc[idx]
        prob = probs[idx]
        
        # Generate reason based on features
        reasons = []
        if food['protein_g'] > 15:
            reasons.append(f"High protein ({food['protein_g']:.1f}g)")
        if food['fiber_g'] > 5:
            reasons.append(f"High fiber ({food['fiber_g']:.1f}g)")
        if food['calories'] < 200:
            reasons.append(f"Low calorie ({food['calories']:.0f} kcal)")
        if food['sugars_g'] < 5:
            reasons.append(f"Low sugar ({food['sugars_g']:.1f}g)")
        if food['cost_per_serving'] < 2:
            reasons.append(f"Budget-friendly (${food['cost_per_serving']:.2f})")
        
        # Dietary flags
        dietary = []
        if food.get('is_vegan', 0) == 1:
            dietary.append('Vegan')
        if food.get('is_glutenfree', 0) == 1:
            dietary.append('Gluten-free')
        if food.get('is_nutfree', 0) == 1:
            dietary.append('Nut-free')
        
        recommendation = {
            'name': food['description'],
            'category': food.get('food_category', 'Unknown'),
            'fit_score': float(prob),
            'confidence': 'high' if prob > 0.8 else 'medium' if prob > 0.6 else 'moderate',
            'nutrition': {
                'calories': float(food['calories']),
                'protein': float(food['protein_g']),
                'carbs': float(food['carbs_g']),
                'fat': float(food['fat_g']),
                'fiber': float(food['fiber_g']),
                'sugars': float(food['sugars_g'])
            },
            'cost': float(food['cost_per_serving']),
            'reasons': reasons,
            'dietary_info': dietary
        }
        recommendations.append(recommendation)
    
    return {
        'recommendations': recommendations,
        'query': query,
        'total_eligible': len(eligible_foods),
        'model_version': '1.0',
        'user_goal': user_profile.get('primaryGoal', 'General Health')
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
          "fit_score": 0.85,
          "confidence": "high",
          "nutrition": {...},
          "reasons": [...],
          "dietary_info": [...]
        }
      ]
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
                    'dietaryRestrictions': ['Vegan'],
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
        sys.exit(1)

if __name__ == '__main__':
    main()
