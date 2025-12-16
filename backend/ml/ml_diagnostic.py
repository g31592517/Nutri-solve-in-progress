"""
ML Pipeline Diagnostic Tool
Detects circular logic, data leakage, and overfitting in food recommendation system

Author: NutriSolve ML Team
Date: December 2025

Purpose: Diagnose why 205-sample dataset achieves suspiciously high 95% F1-score
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_SEEDS = [0, 7, 42, 99, 1234]
ENGINEERED_PATTERNS = ['density', 'score', 'ratio', 'efficiency', 'per_', 'to_']
HIGH_CORRELATION_THRESHOLD = 0.7
VARIANCE_THRESHOLD = 0.10
SINGLE_FEATURE_IMPORTANCE_THRESHOLD = 0.40


class MLDiagnostic:
    """Comprehensive ML pipeline diagnostic tool"""
    
    def __init__(self, data_path, feature_info_path, label_features):
        self.data_path = Path(data_path)
        self.feature_info_path = Path(feature_info_path)
        self.label_features = label_features
        self.df = None
        self.feature_info = None
        self.results = {}
        
    def load_data(self):
        """Load processed data and feature metadata"""
        print("\n" + "="*80)
        print("ML PIPELINE DIAGNOSTIC REPORT")
        print("="*80)
        
        print("\n[1. DATA INTEGRITY CHECK]")
        print("-"*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        with open(self.feature_info_path, 'r') as f:
            self.feature_info = json.load(f)
        
        # Basic info
        print(f"Dataset shape: {self.df.shape[0]} samples × {self.df.shape[1]} columns")
        print(f"Selected features: {len(self.feature_info['selected_features'])} features")
        
        # Class distribution
        fit_count = self.df['fit'].sum()
        fit_pct = fit_count / len(self.df) * 100
        print(f"Label distribution: fit=1 ({fit_count}, {fit_pct:.1f}%), fit=0 ({len(self.df)-fit_count}, {100-fit_pct:.1f}%)")
        
        # Missing values
        missing = self.df.isnull().sum().sum()
        print(f"Missing values: {missing} total")
        
        # Identify feature types
        all_cols = set(self.df.columns) - {'food_name', 'fit', 'category'}
        
        self.raw_features = [col for col in all_cols 
                            if not any(pattern in col.lower() for pattern in ENGINEERED_PATTERNS)
                            and not col.startswith('is_')
                            and not col.startswith('category_')]
        
        self.engineered_features = [col for col in all_cols 
                                   if any(pattern in col.lower() for pattern in ENGINEERED_PATTERNS)]
        
        self.binary_flags = [col for col in all_cols if col.startswith('is_')]
        self.categorical_encoded = [col for col in all_cols if col.startswith('category_')]
        
        print(f"\nFeature categorization:")
        print(f"  • Raw features ({len(self.raw_features)}): {self.raw_features[:5]}...")
        print(f"  • Engineered features ({len(self.engineered_features)}): {self.engineered_features}")
        print(f"  • Binary flags ({len(self.binary_flags)}): {self.binary_flags}")
        print(f"  • Categorical encoded ({len(self.categorical_encoded)}): {len(self.categorical_encoded)} categories")
        
        self.results['data_integrity'] = {
            'n_samples': len(self.df),
            'n_features': self.df.shape[1],
            'fit_percentage': float(fit_pct),
            'missing_values': int(missing),
            'n_raw_features': len(self.raw_features),
            'n_engineered_features': len(self.engineered_features)
        }
        
    def check_duplicates(self):
        """Check for duplicate and redundant samples"""
        print("\n[2. DUPLICATE & REDUNDANCY ANALYSIS]")
        print("-"*80)
        
        # Exact duplicates
        exact_dupes = self.df.duplicated().sum()
        exact_pct = exact_dupes / len(self.df) * 100
        print(f"Exact duplicate rows: {exact_dupes} ({exact_pct:.1f}%)")
        
        # Functional duplicates (same nutritional profile)
        numeric_cols = self.raw_features + self.engineered_features
        numeric_df = self.df[numeric_cols].round(2)  # Round to avoid floating point issues
        functional_dupes = numeric_df.duplicated().sum()
        functional_pct = functional_dupes / len(self.df) * 100
        print(f"Functional duplicates (same nutrition): {functional_dupes} ({functional_pct:.1f}%)")
        
        if functional_pct > 5:
            print(f"⚠️  WARNING: >5% redundancy detected ({functional_pct:.1f}%)")
        else:
            print(f"✓ Redundancy acceptable (<5%)")
        
        self.results['duplicates'] = {
            'exact_duplicates': int(exact_dupes),
            'functional_duplicates': int(functional_dupes),
            'redundancy_percentage': float(functional_pct)
        }
        
    def analyze_correlations(self):
        """Analyze feature-label correlations"""
        print("\n[3. LABEL-FEATURE CORRELATION ANALYSIS]")
        print("-"*80)
        
        features_to_check = self.raw_features + self.engineered_features
        correlations = {}
        
        for feat in features_to_check:
            if feat in self.df.columns:
                corr, pval = pearsonr(self.df[feat], self.df['fit'])
                correlations[feat] = {
                    'correlation': float(corr),
                    'p_value': float(pval),
                    'abs_correlation': float(abs(corr))
                }
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
        
        print(f"\nTop 10 Feature-Label Correlations:")
        print(f"{'Rank':<6} {'Feature':<30} {'Correlation':<15} {'Abs':<10} {'Risk'}")
        print("-"*80)
        
        high_risk_features = []
        for i, (feat, stats) in enumerate(sorted_corrs[:10], 1):
            corr = stats['correlation']
            abs_corr = stats['abs_correlation']
            
            # Risk assessment
            if abs_corr >= 0.8:
                risk = "CRITICAL"
                high_risk_features.append(feat)
            elif abs_corr >= HIGH_CORRELATION_THRESHOLD:
                risk = "HIGH"
                high_risk_features.append(feat)
            elif abs_corr >= 0.5:
                risk = "MODERATE"
            else:
                risk = "LOW"
            
            print(f"{i:<6} {feat:<30} {corr:>14.4f} {abs_corr:>9.4f} {risk}")
        
        if high_risk_features:
            print(f"\n⚠️  HIGH CORRELATION ALERT: {len(high_risk_features)} features with |corr| ≥ {HIGH_CORRELATION_THRESHOLD}")
            print(f"   Features: {high_risk_features}")
        else:
            print(f"\n✓ No high-risk correlations detected")
        
        self.results['correlations'] = {
            'all_correlations': correlations,
            'high_risk_features': high_risk_features,
            'top_10': [{feat: stats} for feat, stats in sorted_corrs[:10]]
        }
        
        return correlations
        
    def detect_circular_features(self, correlations):
        """Detect circular logic in engineered features"""
        print("\n[4. CIRCULAR LOGIC DETECTION]")
        print("-"*80)
        
        circular_features = []
        
        print("\nAnalyzing engineered features for circular logic:")
        print(f"{'Feature':<30} {'Contains Labeling Criteria':<30} {'|Corr|':<10} {'Status'}")
        print("-"*80)
        
        # Known formulas from preprocess.py
        feature_formulas = {
            'nutrient_density': '(protein + iron + vitamin_c) / (calories + 1)',
            'protein_ratio': 'protein / (calories + 1)',
            'carb_fat_ratio': 'carbs / (fat + 0.001)',
            'energy_density': 'calories / 100',
            'micronutrient_score': 'iron*10 + vitamin_c'
        }
        
        for feat in self.engineered_features:
            if feat in correlations:
                abs_corr = correlations[feat]['abs_correlation']
                
                # Check if formula contains labeling criteria
                formula = feature_formulas.get(feat, 'unknown')
                contains_labeling = any(label_feat in formula for label_feat in self.label_features)
                
                # Circular if high correlation AND contains labeling criteria
                is_circular = abs_corr >= 0.8 and contains_labeling
                
                status = "CIRCULAR" if is_circular else "OK"
                if is_circular:
                    circular_features.append(feat)
                    status = f"⚠️  {status}"
                
                contains_str = "YES" if contains_labeling else "NO"
                print(f"{feat:<30} {contains_str:<30} {abs_corr:>9.4f} {status}")
        
        if circular_features:
            print(f"\n⚠️  CIRCULAR LOGIC DETECTED: {len(circular_features)} features")
            print(f"   These features are mathematical proxies for labeling criteria:")
            for feat in circular_features:
                print(f"   • {feat}: {feature_formulas.get(feat, 'formula unknown')}")
        else:
            print(f"\n✓ No circular logic detected in engineered features")
        
        self.results['circular_features'] = {
            'detected': circular_features,
            'count': len(circular_features),
            'formulas': {feat: feature_formulas.get(feat, 'unknown') for feat in circular_features}
        }
        
        return circular_features
        
    def performance_decomposition(self):
        """Compare model performance with different feature sets"""
        print("\n[5. PERFORMANCE DECOMPOSITION EXPERIMENT]")
        print("-"*80)
        
        selected_features = self.feature_info['selected_features']
        
        # Filter features to only those available in processed_data.csv
        available_features = [f for f in selected_features if f in self.df.columns]
        unavailable_features = [f for f in selected_features if f not in self.df.columns]
        
        if unavailable_features:
            print(f"\nNote: {len(unavailable_features)} features from training not in processed_data.csv")
            print(f"      (These are category-encoded features created during preprocessing)")
            print(f"      Using {len(available_features)} available features for analysis")
        
        # Define feature sets
        raw_only = [f for f in available_features if f in self.raw_features]
        no_circular = [f for f in available_features if f not in self.results['circular_features']['detected']]
        
        print(f"\nFeature set definitions:")
        print(f"  • RAW: {len(raw_only)} features (only original nutritional data)")
        print(f"  • FULL: {len(available_features)} features (raw + engineered)")
        print(f"  • NO_CIRCULAR: {len(no_circular)} features (excluding circular features)")
        print(f"  • BASELINE: Majority class prediction")
        
        # Prepare data
        X_full = self.df[available_features].values
        X_raw = self.df[raw_only].values if raw_only else np.zeros((len(self.df), 1))
        X_no_circular = self.df[no_circular].values
        y = self.df['fit'].values
        
        # Baseline accuracy (majority class)
        baseline_acc = max(y.mean(), 1 - y.mean())
        
        results_by_seed = {}
        
        print(f"\nRunning 10-fold CV across {len(RANDOM_SEEDS)} random seeds...")
        print(f"{'Seed':<8} {'Model':<15} {'F1 (mean±std)':<20} {'AUC':<15} {'Accuracy'}")
        print("-"*80)
        
        for seed in RANDOM_SEEDS:
            seed_results = {}
            
            # 10-fold CV with stratification
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            
            for model_name, X_data in [('RAW', X_raw), ('FULL', X_full), ('NO_CIRCULAR', X_no_circular)]:
                if X_data.shape[1] == 0 or (X_data.shape[1] == 1 and X_data.sum() == 0):
                    continue
                    
                # Train model
                clf = RandomForestClassifier(
                    n_estimators=50, 
                    max_depth=5, 
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=seed
                )
                
                # Cross-validation scores
                f1_scores = cross_val_score(clf, X_data, y, cv=cv, scoring='f1_macro')
                auc_scores = cross_val_score(clf, X_data, y, cv=cv, scoring='roc_auc')
                acc_scores = cross_val_score(clf, X_data, y, cv=cv, scoring='accuracy')
                
                f1_mean, f1_std = f1_scores.mean(), f1_scores.std()
                auc_mean = auc_scores.mean()
                acc_mean = acc_scores.mean()
                
                seed_results[model_name] = {
                    'f1_mean': float(f1_mean),
                    'f1_std': float(f1_std),
                    'f1_scores': f1_scores.tolist(),
                    'auc_mean': float(auc_mean),
                    'accuracy_mean': float(acc_mean)
                }
                
                print(f"{seed:<8} {model_name:<15} {f1_mean:.4f}±{f1_std:.4f}      {auc_mean:>14.4f} {acc_mean:>14.4f}")
            
            results_by_seed[seed] = seed_results
        
        # Aggregate across seeds
        print(f"\n{'Model':<15} {'F1 (mean±std)':<20} {'Min F1':<12} {'Max F1':<12} {'Variance'}")
        print("-"*80)
        
        model_aggregates = {}
        for model_name in ['RAW', 'FULL', 'NO_CIRCULAR']:
            all_f1_means = [results_by_seed[seed][model_name]['f1_mean'] 
                           for seed in RANDOM_SEEDS if model_name in results_by_seed[seed]]
            
            if all_f1_means:
                f1_mean = np.mean(all_f1_means)
                f1_std = np.std(all_f1_means)
                f1_min = np.min(all_f1_means)
                f1_max = np.max(all_f1_means)
                
                model_aggregates[model_name] = {
                    'f1_mean': float(f1_mean),
                    'f1_std': float(f1_std),
                    'f1_min': float(f1_min),
                    'f1_max': float(f1_max)
                }
                
                print(f"{model_name:<15} {f1_mean:.4f}±{f1_std:.4f}      {f1_min:>11.4f} {f1_max:>11.4f} {f1_std:>11.4f}")
        
        print(f"{'BASELINE':<15} {baseline_acc:.4f}±0.0000      {baseline_acc:>11.4f} {baseline_acc:>11.4f} {0.0:>11.4f}")
        
        # Check for suspicious jumps
        if 'RAW' in model_aggregates and 'FULL' in model_aggregates:
            improvement = model_aggregates['FULL']['f1_mean'] - model_aggregates['RAW']['f1_mean']
            print(f"\nFULL vs RAW improvement: {improvement:.4f} ({improvement*100:.1f}%)")
            if improvement > 0.20:
                print(f"⚠️  WARNING: >20% improvement suspicious for engineered features")
        
        self.results['performance_decomposition'] = {
            'by_seed': results_by_seed,
            'aggregates': model_aggregates,
            'baseline_accuracy': float(baseline_acc)
        }
        
        return model_aggregates
        
    def test_cross_validation_stability(self):
        """Test stability across multiple train-test splits"""
        print("\n[6. CROSS-VALIDATION STABILITY TEST]")
        print("-"*80)
        
        selected_features = self.feature_info['selected_features']
        available_features = [f for f in selected_features if f in self.df.columns]
        X = self.df[available_features].values
        y = self.df['fit'].values
        
        f1_scores = []
        auc_scores = []
        
        print(f"Running 10 different 80/20 splits with different random seeds...")
        print(f"{'Split':<8} {'Train Fit%':<15} {'Test Fit%':<15} {'F1-macro':<12} {'ROC-AUC'}")
        print("-"*80)
        
        for i in range(10):
            seed = i * 111  # Different seeds for each split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            
            # Check test set has sufficient minority samples
            minority_test = min(y_test.sum(), len(y_test) - y_test.sum())
            if minority_test < 5:
                print(f"Split {i+1:<3} ⚠️  WARNING: Only {minority_test} minority samples in test set")
                continue
            
            # Train model
            clf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=5, 
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            f1 = f1_score(y_test, y_pred, average='macro')
            auc = roc_auc_score(y_test, y_proba)
            
            f1_scores.append(f1)
            auc_scores.append(auc)
            
            train_fit_pct = y_train.mean() * 100
            test_fit_pct = y_test.mean() * 100
            
            print(f"Split {i+1:<3} {train_fit_pct:>14.1f}% {test_fit_pct:>14.1f}% {f1:>11.4f} {auc:>11.4f}")
        
        # Calculate stability metrics
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        f1_min = np.min(f1_scores)
        f1_max = np.max(f1_scores)
        f1_range = f1_max - f1_min
        
        print(f"\nStability Analysis:")
        print(f"  F1 mean: {f1_mean:.4f}")
        print(f"  F1 std:  {f1_std:.4f} (variance)")
        print(f"  F1 range: [{f1_min:.4f}, {f1_max:.4f}] (spread: {f1_range:.4f})")
        
        if f1_std > VARIANCE_THRESHOLD:
            print(f"  ⚠️  UNSTABLE: Variance {f1_std:.4f} > {VARIANCE_THRESHOLD} threshold")
            print(f"     Results highly dependent on train-test split luck")
        else:
            print(f"  ✓ STABLE: Variance {f1_std:.4f} ≤ {VARIANCE_THRESHOLD} threshold")
        
        self.results['stability'] = {
            'f1_scores': [float(x) for x in f1_scores],
            'f1_mean': float(f1_mean),
            'f1_std': float(f1_std),
            'f1_min': float(f1_min),
            'f1_max': float(f1_max),
            'is_stable': bool(f1_std <= VARIANCE_THRESHOLD)
        }
        
        return f1_std
        
    def permutation_importance_test(self):
        """Test feature importance via permutation"""
        print("\n[7. PERMUTATION IMPORTANCE TEST]")
        print("-"*80)
        
        selected_features = self.feature_info['selected_features']
        available_features = [f for f in selected_features if f in self.df.columns]
        X = self.df[available_features].values
        y = self.df['fit'].values
        
        # Single train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        clf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5, 
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Baseline performance
        baseline_f1 = f1_score(y_test, clf.predict(X_test), average='macro')
        
        # Permutation importance
        perm_importance = permutation_importance(
            clf, X_test, y_test, 
            n_repeats=10, 
            random_state=42, 
            scoring='f1_macro'
        )
        
        print(f"\nPermutation Importance (F1-macro drop when feature shuffled):")
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':<15} {'Std':<10} {'% Drop'}")
        print("-"*80)
        
        # Sort by importance
        importance_data = []
        for i, feat in enumerate(available_features):
            importance = perm_importance.importances_mean[i]
            std = perm_importance.importances_std[i]
            pct_drop = (importance / baseline_f1) * 100 if baseline_f1 > 0 else 0
            
            importance_data.append({
                'feature': feat,
                'importance': float(importance),
                'std': float(std),
                'pct_drop': float(pct_drop)
            })
        
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        critical_features = []
        for i, data in enumerate(importance_data, 1):
            feat = data['feature']
            imp = data['importance']
            std = data['std']
            pct = data['pct_drop']
            
            if pct > 50:
                critical_features.append(feat)
                status = "⚠️  CRITICAL"
            else:
                status = ""
            
            print(f"{i:<6} {feat:<30} {imp:>14.4f} {std:>9.4f} {pct:>9.1f}% {status}")
        
        if critical_features:
            print(f"\n⚠️  CRITICAL FEATURES: {len(critical_features)} features cause >50% F1 drop")
            print(f"   These features may contain labeling logic:")
            for feat in critical_features:
                print(f"   • {feat}")
        
        self.results['permutation_importance'] = {
            'baseline_f1': float(baseline_f1),
            'importances': importance_data,
            'critical_features': critical_features
        }
        
        return critical_features
        
    def generate_recommendation(self):
        """Generate final diagnostic recommendation"""
        print("\n" + "="*80)
        print("[8. FINAL DIAGNOSTIC RECOMMENDATION]")
        print("="*80)
        
        # Gather evidence
        circular_detected = len(self.results['circular_features']['detected']) > 0
        high_variance = not self.results['stability']['is_stable']
        
        perf_decomp = self.results['performance_decomposition']['aggregates']
        full_f1 = perf_decomp.get('FULL', {}).get('f1_mean', 0)
        raw_f1 = perf_decomp.get('RAW', {}).get('f1_mean', 0)
        improvement = full_f1 - raw_f1
        
        high_correlation_features = self.results['correlations']['high_risk_features']
        critical_perm_features = self.results.get('permutation_importance', {}).get('critical_features', [])
        
        # Decision logic
        recommendation = None
        reasoning = []
        
        if circular_detected:
            recommendation = "remove_engineered_from_training"
            reasoning.append(f"✗ {len(self.results['circular_features']['detected'])} circular features detected")
            reasoning.append(f"  Features are mathematical proxies for labeling criteria")
            
        if high_variance:
            if not recommendation:
                recommendation = "collect_more_data"
            reasoning.append(f"✗ Unstable results (F1 variance = {self.results['stability']['f1_std']:.4f} > {VARIANCE_THRESHOLD})")
            reasoning.append(f"  Performance highly dependent on train-test split")
            
        if improvement > 0.20:
            if not recommendation:
                recommendation = "simplify_model"
            reasoning.append(f"✗ Suspicious improvement (FULL vs RAW: {improvement:.4f} = {improvement*100:.1f}%)")
            reasoning.append(f"  Engineered features provide unrealistic boost")
            
        if len(critical_perm_features) > 0:
            reasoning.append(f"✗ {len(critical_perm_features)} features cause >50% F1 drop when shuffled")
            reasoning.append(f"  Model relies too heavily on single features")
            
        if len(high_correlation_features) > 0:
            reasoning.append(f"⚠️  {len(high_correlation_features)} features with |corr| ≥ {HIGH_CORRELATION_THRESHOLD}")
            
        if not recommendation:
            recommendation = "ok"
            reasoning = [
                "✓ No circular features detected",
                f"✓ Stable results (F1 variance = {self.results['stability']['f1_std']:.4f} ≤ {VARIANCE_THRESHOLD})",
                f"✓ Reasonable improvement (FULL vs RAW: {improvement*100:.1f}%)",
                "✓ No single feature dominance"
            ]
        
        # Print recommendation
        print(f"\nRECOMMENDATION: {recommendation.upper()}")
        print(f"\nReasoning:")
        for reason in reasoning:
            print(f"  {reason}")
        
        # Detailed action items
        print(f"\nAction Items:")
        if recommendation == "remove_engineered_from_training":
            print(f"  1. Remove circular engineered features from training:")
            for feat in self.results['circular_features']['detected']:
                print(f"     - {feat}")
            print(f"  2. Re-train model with RAW features only")
            print(f"  3. Verify F1 drops to realistic ~70-80% range")
            print(f"  4. If still high, investigate labeling methodology")
            
        elif recommendation == "collect_more_data":
            print(f"  1. Collect more food samples (target: 500-1000)")
            print(f"  2. Ensure diverse nutritional profiles")
            print(f"  3. Re-run diagnostics to verify stability")
            
        elif recommendation == "adjust_labeling":
            print(f"  1. Review percentile-based labeling thresholds")
            print(f"  2. Consider domain expert labeling instead")
            print(f"  3. Validate labels against nutritionist guidelines")
            
        elif recommendation == "simplify_model":
            print(f"  1. Use simpler model (logistic regression)")
            print(f"  2. Reduce feature complexity")
            print(f"  3. Add regularization")
            
        else:  # ok
            print(f"  1. Proceed with current pipeline")
            print(f"  2. Monitor performance on new data")
            print(f"  3. Conduct regular diagnostic checks")
        
        self.results['recommendation'] = {
            'action': recommendation,
            'reasoning': reasoning,
            'evidence': {
                'circular_features': bool(circular_detected),
                'unstable_results': bool(high_variance),
                'suspicious_improvement': bool(improvement > 0.20),
                'critical_features': bool(len(critical_perm_features) > 0)
            }
        }
        
        return recommendation
        
    def save_report(self, output_path):
        """Save diagnostic report to JSON"""
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Diagnostic report saved to: {output_path}")
        
    def run_full_diagnostic(self, output_path):
        """Run complete diagnostic pipeline"""
        self.load_data()
        self.check_duplicates()
        correlations = self.analyze_correlations()
        self.detect_circular_features(correlations)
        self.performance_decomposition()
        self.test_cross_validation_stability()
        self.permutation_importance_test()
        self.generate_recommendation()
        self.save_report(output_path)
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='ML Pipeline Diagnostic Tool')
    parser.add_argument('--data-path', default='processed_data.csv', help='Path to processed data CSV')
    parser.add_argument('--feature-info', default='feature_info.json', help='Path to feature info JSON')
    parser.add_argument('--label-features', default='protein,fat,calories,iron,vitamin_c', 
                       help='Comma-separated list of labeling features')
    parser.add_argument('--output', default='diagnostic_report.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    # Parse label features
    label_features = [f.strip() for f in args.label_features.split(',')]
    
    # Run diagnostic
    diagnostic = MLDiagnostic(args.data_path, args.feature_info, label_features)
    diagnostic.run_full_diagnostic(args.output)


if __name__ == '__main__':
    main()
