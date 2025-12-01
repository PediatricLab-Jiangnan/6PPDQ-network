# =============================================================================
# Machine Learning Modeling
# =============================================================================

# ==================== Import Libraries ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import (
    Lasso, Ridge, LogisticRegression, ElasticNet
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# ==================== 1. Load and Preprocess Data ====================
print("\n" + "="*60)
print("[Step 1] Loading and preprocessing dataset")
print("="*60)

# 1.1 Load data
print("\n  -> Reading ML_dataset.csv...")
ml_data = pd.read_csv('ML_dataset.csv', encoding='utf-8-sig')
print(f"     ✓ Loaded successfully. Shape: {ml_data.shape}")

# 1.2 Handle index column
first_col = ml_data.columns[0]
if first_col == '' or 'Unnamed' in first_col:
    ml_data = ml_data.rename(columns={first_col: 'Sample_ID'})
ml_data = ml_data.set_index('Sample_ID')
print(f"     Processed shape: {ml_data.shape}")

# 1.3 Check class distribution
print("\n  -> Checking group distribution...")
group_counts = ml_data['Group'].value_counts()
for group, count in group_counts.items():
    print(f"       - {group}: {count} samples")

# 1.4 Separate features (X) and labels (y)
y = ml_data['Group']
X = ml_data.drop(columns=['Group'])
print(f"\n  -> Features: {X.shape[1]} genes, Samples: {X.shape[0]}")

# 1.5 Handle missing values
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"     ⚠️  Filling {missing_count} missing values with column means")
    X = X.fillna(X.mean())
else:
    print("     ✓ No missing values")

# 1.6 Encode binary labels: 'Seizures' → 1, others → 0
y_encoded = (y == 'Seizures').astype(int)
print(f"\n  -> Label encoding: 0={sum(y_encoded==0)}, 1={sum(y_encoded==1)}")

# 1.7 Stratified train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n  -> Train/Test split: {X_train.shape[0]} / {X_test.shape[0]} samples")

# 1.8 Standardize features (Z-score)
scaler = StandardScaler()
X_train_scaled_array = scaler.fit_transform(X_train)
X_test_scaled_array = scaler.transform(X_test)

# Keep both raw and scaled versions for flexibility
X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)
print("     ✓ Standardization completed")

# Save global metadata
total_samples = len(ml_data)
total_features = X.shape[1]

print("\n" + "="*60)
print("[Data Preparation Complete] Summary")
print("="*60)
print(f"  Total samples: {total_samples}")
print(f"  Total genes: {total_features}")
print(f"  Train: {X_train.shape[0]} × {X_train.shape[1]}")
print(f"  Test:  {X_test.shape[0]} × {X_test.shape[1]}")
print("="*60 + "\n")

# ==================== 2. Define Feature Selection Algorithms (12 Methods) ====================
print("\n[Step 2] Defining 12 feature selection algorithms...")

feature_selection_algorithms = {
    'Lasso': Lasso(alpha=0.01, random_state=123, max_iter=5000),
    'Ridge': Ridge(alpha=1.0, random_state=123),
    'Stepglm': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=123),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=123, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    'RF': RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1),
    'Enet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=123, max_iter=5000),
    'plsRglm': Ridge(alpha=0.5, random_state=123),  # Placeholder for PLS-like behavior
    'GBM': GradientBoostingClassifier(n_estimators=100, random_state=123),
    'NaiveBayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'glmBoost': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=123),
    'SVM': SVC(kernel='linear', random_state=123),
}

print(f"✓ Defined {len(feature_selection_algorithms)} feature selectors")

# ==================== 3. Perform Feature Selection ====================
print("\n[Step 3] Executing feature selection (top 50 features per method)...")

feature_selection_results = {}

for i, (name, _) in enumerate(feature_selection_algorithms.items(), 1):
    print(f"  [{i:2d}/12] {name:15s}", end=" ")
    start_time = time.time()
    
    try:
        if name in ['Lasso', 'Ridge', 'Enet', 'plsRglm']:
            # Linear models: use absolute coefficients
            model = clone(feature_selection_algorithms[name])
            model.fit(X_train_scaled, y_train)
            coef = np.abs(model.coef_).flatten()
            top_50_idx = np.argsort(coef)[-50:]
            selected_features = X_train.columns[top_50_idx].tolist()
        
        elif name in ['XGBoost', 'RF', 'GBM', 'glmBoost']:
            # Tree-based: use feature_importances_
            model = clone(feature_selection_algorithms[name])
            # Reduce dimensionality if too many features (>1000)
            if X_train.shape[1] > 1000:
                variances = X_train.var()
                top_1000_idx = np.argsort(variances)[-1000:]
                X_train_reduced = X_train.iloc[:, top_1000_idx]
            else:
                X_train_reduced = X_train
            model.fit(X_train_reduced, y_train)
            importances = model.feature_importances_
            top_50_idx = np.argsort(importances)[-50:]
            selected_features = X_train_reduced.columns[top_50_idx].tolist()
        
        elif name == 'Stepglm':
            # L1 Logistic Regression with variance pre-filtering
            model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=123, max_iter=1000)
            if X_train.shape[1] > 500:
                variances = X_train.var()
                top_500_idx = np.argsort(variances)[-500:]
                X_train_reduced = X_train_scaled.iloc[:, top_500_idx]
            else:
                X_train_reduced = X_train_scaled
            model.fit(X_train_reduced, y_train)
            coef = np.abs(model.coef_).flatten()
            top_50_idx = np.argsort(coef)[-50:]
            selected_features = X_train_reduced.columns[top_50_idx].tolist()
        
        elif name == 'LDA':
            # LDA requires n_features < n_samples
            n_samples = X_train.shape[0]
            if X_train.shape[1] >= n_samples:
                variances = X_train.var()
                top_idx = np.argsort(variances)[-(n_samples - 1):]
                X_train_reduced = X_train_scaled.iloc[:, top_idx]
            else:
                X_train_reduced = X_train_scaled
            model = LinearDiscriminantAnalysis()
            model.fit(X_train_reduced, y_train)
            coef = np.abs(model.coef_).flatten()
            top_k = min(50, len(coef))
            top_50_idx = np.argsort(coef)[-top_k:]
            selected_features = X_train_reduced.columns[top_50_idx].tolist()
        
        elif name == 'SVM':
            # Linear SVM: use |coef_|
            if X_train.shape[1] > 500:
                variances = X_train.var()
                top_500_idx = np.argsort(variances)[-500:]
                X_train_reduced = X_train_scaled.iloc[:, top_500_idx]
            else:
                X_train_reduced = X_train_scaled
            model = SVC(kernel='linear', random_state=123)
            model.fit(X_train_reduced, y_train)
            coef = np.abs(model.coef_).flatten()
            top_50_idx = np.argsort(coef)[-50:]
            selected_features = X_train_reduced.columns[top_50_idx].tolist()
        
        elif name == 'NaiveBayes':
            # Use top 50 by variance (no intrinsic weights)
            variances = X_train.var()
            top_50_idx = np.argsort(variances)[-50:]
            selected_features = X_train.columns[top_50_idx].tolist()
        
        feature_selection_results[name] = selected_features
        elapsed = time.time() - start_time
        print(f"-> ✓ {len(selected_features):3d} features ({elapsed:.1f}s)")
    
    except Exception as e:
        print(f"-> ✗ Failed: {str(e)[:30]}")
        # Fallback: use top 50 by variance
        variances = X_train.var()
        top_50_idx = np.argsort(variances)[-50:]
        feature_selection_results[name] = X_train.columns[top_50_idx].tolist()
        print(f"       Using variance-based fallback: {len(feature_selection_results[name])} features")

# Print summary
print("\n" + "=" * 70)
print("Feature Selection Summary:")
print("=" * 70)
print(f"{'Method':<20s} | {'#Features':>8s}")
print("-" * 70)
for method, features in feature_selection_results.items():
    print(f"{method:<20s} | {len(features):>8d}")
print("=" * 70)

# ==================== 4. Define Classification Models (9 Methods) ====================
print("\n[Step 4] Defining 9 classification algorithms...")

modeling_methods = {
    'LDA': LinearDiscriminantAnalysis(),
    'Ridge': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=123),
    'Lasso_Model': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=123),
    'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=1000, random_state=123),
    'RF_Model': RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=123),
    'GBM': GradientBoostingClassifier(n_estimators=100, random_state=123),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=123, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    'NaiveBayes': GaussianNB(),
}

print(f"✓ Defined {len(modeling_methods)} classifiers")

# ==================== 5. Generate Unique Combinations (Avoid Duplicates) ====================
print("\n[Step 5] Generating unique modeling combinations...")

all_combinations = []

# Mapping to avoid redundant FS+Model pairs (e.g., RF+RF_Model = duplicate of standalone RF)
duplicate_mapping = {
    'RF': 'RF_Model',
    'GBM': 'GBM',
    'XGBoost': 'XGBoost',
    'NaiveBayes': 'NaiveBayes',
    'LDA': 'LDA',
    'SVM': 'SVM',
    'Lasso': 'Lasso_Model',
    'Ridge': 'Ridge',
}

# Type 1: Feature Selection + Modeling (exclude duplicates)
combined_count = 0
skipped_duplicates = []
for fs_method in feature_selection_results:
    for model_name in modeling_methods:
        if fs_method in duplicate_mapping and duplicate_mapping[fs_method] == model_name:
            skipped_duplicates.append(f"{fs_method}+{model_name}")
            continue
        all_combinations.append((f"{fs_method}+{model_name}", fs_method, model_name, 'combined'))
        combined_count += 1

print(f"  -> Combined (FS + Model): {combined_count} (skipped {len(skipped_duplicates)} duplicates)")

# Type 2: Standalone FS methods that are also classifiers
classification_fs_methods = {'XGBoost', 'RF', 'GBM', 'glmBoost', 'NaiveBayes', 'LDA', 'SVM', 'Stepglm'}
standalone_fs_list = []
for fs in feature_selection_results:
    if fs in classification_fs_methods:
        all_combinations.append((fs, fs, None, 'fs_only'))
        standalone_fs_list.append(fs)
print(f"  -> Standalone FS classifiers: {len(standalone_fs_list)}")

# Type 3: Standalone models using all features (avoid overlap with standalone FS)
model_to_fs = {
    'RF_Model': 'RF',
    'GBM': 'GBM',
    'XGBoost': 'XGBoost',
    'NaiveBayes': 'NaiveBayes',
    'LDA': 'LDA',
    'SVM': 'SVM'
}
kept_models = []
for model in modeling_methods:
    if model in classification_fs_methods:
        continue  # Already covered as standalone FS
    if model in model_to_fs and model_to_fs[model] in classification_fs_methods:
        continue  # Redundant
    all_combinations.append((model, 'AllFeatures', model, 'model_only'))
    kept_models.append(model)
print(f"  -> Standalone full-feature models: {len(kept_models)}")

print(f"\nTotal unique combinations: {len(all_combinations)}")

# ==================== 6. Train Models and Evaluate AUC ====================
print("\n[Step 6] Training models and computing AUC scores...")

results = []
successful, skipped, failed = 0, 0, 0

for i, (combo_name, fs_method, model_name, combo_type) in enumerate(all_combinations, 1):
    print(f"[{i:3d}/{len(all_combinations)}] {combo_name:45s}", end=" ")
    
    try:
        if combo_type == 'combined':
            features = feature_selection_results[fs_method]
            if len(features) < 3:
                print(f"-> Skipped (too few features: {len(features)})")
                skipped += 1
                continue
            X_tr, X_te = X_train[features], X_test[features]
            model = clone(modeling_methods[model_name])
            model.fit(X_tr, y_train)
            y_tr_pred = model.predict_proba(X_tr)[:, 1]
            y_te_pred = model.predict_proba(X_te)[:, 1]
        
        elif combo_type == 'fs_only':
            features = feature_selection_results[fs_method]
            if len(features) < 3:
                print(f"-> Skipped (too few features: {len(features)})")
                skipped += 1
                continue
            X_tr, X_te = X_train[features], X_test[features]
            # Re-instantiate classifier matching the FS method
            if fs_method == 'XGBoost':
                model = XGBClassifier(n_estimators=100, random_state=123, use_label_encoder=False, eval_metric='logloss', verbosity=0)
            elif fs_method == 'RF':
                model = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
            elif fs_method == 'GBM':
                model = GradientBoostingClassifier(n_estimators=100, random_state=123)
            elif fs_method == 'glmBoost':
                model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=123)
            elif fs_method == 'NaiveBayes':
                model = GaussianNB()
            elif fs_method == 'LDA':
                model = LinearDiscriminantAnalysis()
            elif fs_method == 'SVM':
                model = SVC(kernel='rbf', probability=True, random_state=123)
            elif fs_method == 'Stepglm':
                model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=123)
            model.fit(X_tr, y_train)
            y_tr_pred = model.predict_proba(X_tr)[:, 1]
            y_te_pred = model.predict_proba(X_te)[:, 1]
        
        elif combo_type == 'model_only':
            X_tr, X_te = X_train, X_test
            model = clone(modeling_methods[model_name])
            model.fit(X_tr, y_train)
            y_tr_pred = model.predict_proba(X_tr)[:, 1]
            y_te_pred = model.predict_proba(X_te)[:, 1]
        
        # Compute AUC
        train_auc = roc_auc_score(y_train, y_tr_pred)
        test_auc = roc_auc_score(y_test, y_te_pred)
        results.append({
            'Method': combo_name,
            'FeatureSelection': fs_method,
            'Model': model_name or fs_method,
            'Type': combo_type,
            'N_Features': X_tr.shape[1],
            'Train_AUC': train_auc,
            'Test_AUC': test_auc,
            'Mean_AUC': (train_auc + test_auc) / 2
        })
        successful += 1
        print(f"-> ✓ Train={train_auc:.3f}, Test={test_auc:.3f}")
    
    except Exception as e:
        print(f"-> ✗ Error: {str(e)[:50]}")
        failed += 1

print(f"\n✓ Success: {successful}, ⚠ Skipped: {skipped}, ✗ Failed: {failed}")

# ==================== 7. Save Results ====================
if not results:
    raise RuntimeError("No models succeeded. Exiting.")

results_df = pd.DataFrame(results).sort_values('Mean_AUC', ascending=False)

results_df.to_csv('03_all_models_results.csv', index=False)
results_df[['Method', 'Train_AUC', 'Test_AUC']].set_index('Method').to_csv(
    '04_AUC_matrix_for_R.txt', sep='\t'
)
results_df.head(10).to_csv('05_top10_models.csv', index=False)

with open('01_feature_selection_results.txt', 'w', encoding='utf-8') as f:
    for method, feats in feature_selection_results.items():
        f.write(f"{method}\t{','.join(feats)}\n")

print("\n✓ Results saved successfully.")

# ==================== 8. Visualization ====================
plt.figure(figsize=(8, max(10, min(30, len(results_df)) * 0.3)))
top30 = results_df.head(30)
heatmap_data = top30.set_index('Method')[['Train_AUC', 'Test_AUC']]
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
            cbar_kws={'label': 'AUC'})
plt.title('Top 30 Models AUC Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Cohort')
plt.ylabel('Method')
plt.tight_layout()
plt.savefig('06_python_preview_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Heatmap saved: 06_python_preview_heatmap.png")

# ==================== 9. Final Report ====================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE — SUMMARY")
print("=" * 80)
print(f"Samples: {total_samples} | Genes: {total_features}")
print(f"Combinations: {len(all_combinations)} → Successful: {successful}")
print(f"Top Mean AUC: {results_df.iloc[0]['Mean_AUC']:.4f} ({results_df.iloc[0]['Method']})")

type_map = {'combined': 'FS+Model', 'fs_only': 'Standalone FS', 'model_only': 'Full-Feature Model'}
print("\nBreakdown by type:")
for t, count in results_df['Type'].value_counts().items():
    print(f"  {type_map.get(t, t)}: {count}")

print("\nTop 5 Models:")
print(results_df[['Method', 'Mean_AUC', 'Test_AUC']].head().to_string(index=False))

print("\n✅ Pipeline finished. Proceed to R for advanced visualization.")
print("Generated files:")
print("  • 01_feature_selection_results.txt")
print("  • 03_all_models_results.csv")
print("  • 04_AUC_matrix_for_R.txt  ← Use this in R")
print("  • 05_top10_models.csv")
print("  • 06_python_preview_heatmap.png")
print("=" * 80)
