# ==================== Import Required Libraries ====================
import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# Feature Selection & Modeling Algorithms
from sklearn.linear_model import (
    Lasso, Ridge, LogisticRegression, ElasticNet
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier  # Added: AdaBoost
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # Added: Decision Tree
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 80)
print("Machine Learning Modeling Pipeline - 113 Combinations")
print("=" * 80)

# ==================== Configuration Parameters ====================
RANDOM_SEED = 123
TEST_SIZE = 0.3      # 30% of data reserved for testing
MAX_ITER = 5000      # Max iterations for linear models

# ==================== Step 1: Load and Preprocess Data ====================
def load_and_preprocess_data(filepath: str):
    """
    Load dataset from CSV, perform basic preprocessing, and split into train/test sets.
    
    Returns:
        X_train, X_test: Original feature matrices (unscaled)
        X_train_scaled, X_test_scaled: Scaled feature matrices
        y_train, y_test: Encoded binary labels
    """
    print("\n" + "=" * 60)
    print("[Step 1] Loading and Preprocessing Data")
    print("=" * 60)

    # 1.1 Load data
    print(f"  -> Reading file: {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"  ❌ Error: File not found: {filepath}")
        return None, None, None, None, None, None

    # 1.2 Handle index column (often unnamed in exported CSVs)
    first_col = df.columns[0]
    if first_col == '' or 'Unnamed' in first_col:
        df = df.rename(columns={first_col: 'Sample_ID'})
    if 'Sample_ID' in df.columns:
        df = df.set_index('Sample_ID')

    # 1.3 Validate presence of target column
    if 'Group' not in df.columns:
        raise ValueError("Dataset must contain a 'Group' column as the target.")

    print(f"     Original shape: {df.shape}")
    print(f"     Class distribution:")
    print(df['Group'].value_counts())

    # 1.4 Separate features (X) and target (y)
    y = df['Group']
    X = df.drop(columns=['Group'])

    # 1.5 Impute missing values with column-wise mean
    if X.isnull().sum().sum() > 0:
        print(f"     ⚠️ Missing values detected → imputed with column means.")
        X = X.fillna(X.mean())

    # 1.6 Encode labels: assume 'Seizures' is positive class; otherwise use first unique class
    unique_classes = y.unique()
    pos_class = 'Seizures' if 'Seizures' in unique_classes else unique_classes[0]
    y_encoded = (y == pos_class).astype(int)
    print(f"     Label encoding: '{pos_class}' → 1, others → 0")

    # 1.7 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=42,  # Fixed seed for reproducibility
        stratify=y_encoded
    )

    # 1.8 Standardize features using Z-score (fit only on training set)
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    # Convert back to DataFrame with original column/index info
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)

    print(f"     ✓ Standardization completed.")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

# ==================== Step 2: Define Feature Selection Methods (12 total) ====================
def get_feature_selectors():
    """Return a dictionary of 12 feature selection methods."""
    return {
        'Lasso': Lasso(alpha=0.01, random_state=RANDOM_SEED, max_iter=MAX_ITER),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
        'Stepglm': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_SEED),  # Simulates stepwise GLM
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'RF': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'Enet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=MAX_ITER),
        'plsRglm': Ridge(alpha=0.5, random_state=RANDOM_SEED),  # Simplified PLS surrogate
        'GBM': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'NaiveBayes': GaussianNB(),  # Proxy: uses variance-based ranking
        'LDA': LinearDiscriminantAnalysis(),
        'glmBoost': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=RANDOM_SEED),
        'SVM': SVC(kernel='linear', random_state=RANDOM_SEED)
    }

# ==================== Step 3: Define Classification Models (11 total) ====================
def get_classifiers():
    """Return a dictionary of 11 classification algorithms."""
    return {
        'LDA': LinearDiscriminantAnalysis(),
        'Ridge': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        'Lasso': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_SEED),
        'glmnet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        'RF': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED),
        'GBM': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'NaiveBayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED),
        'DT': DecisionTreeClassifier(random_state=RANDOM_SEED)
    }

# ==================== Step 4: Run Feature Selection ====================
def run_feature_selection(X_train, X_train_scaled, y_train):
    """
    Apply 12 feature selection methods and return top ~50 features per method.
    
    Returns:
        dict: {method_name: list_of_selected_features}
    """
    print("\n[Step 2] Performing Feature Selection (12 Methods)...")

    selectors = get_feature_selectors()
    fs_results = {}

    for name, model in selectors.items():
        start_time = time.time()
        selected_feats = []

        try:
            # A. Linear models: select based on |coefficient|
            if name in ['Lasso', 'Ridge', 'Enet', 'plsRglm']:
                model.fit(X_train_scaled, y_train)
                coef = np.abs(model.coef_).flatten()
                top_idx = np.argsort(coef)[-50:]
                if name in ['Lasso', 'Enet']:
                    real_top = [i for i in top_idx if coef[i] > 1e-5]
                    if len(real_top) < 2:
                        real_top = top_idx
                    top_idx = real_top
                selected_feats = X_train.columns[top_idx].tolist()

            # B. Tree-based ensembles: use feature_importances_
            elif name in ['XGBoost', 'RF', 'GBM', 'glmBoost']:
                X_curr = X_train
                if X_train.shape[1] > 2000:
                    vars_idx = np.argsort(X_train.var())[-2000:]
                    X_curr = X_train.iloc[:, vars_idx]
                model.fit(X_curr, y_train)
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-50:]
                selected_feats = X_curr.columns[top_idx].tolist()

            # C. Stepglm (L1 logistic regression)
            elif name == 'Stepglm':
                model.fit(X_train_scaled, y_train)
                coef = np.abs(model.coef_).flatten()
                top_idx = np.argsort(coef)[-50:]
                selected_feats = X_train.columns[top_idx].tolist()

            # D. SVM / LDA: use |coef_| for linear kernel
            elif name in ['SVM', 'LDA']:
                model.fit(X_train_scaled, y_train)
                coef = np.abs(model.coef_).flatten()
                top_idx = np.argsort(coef)[-50:]
                selected_feats = X_train.columns[top_idx].tolist()

            # E. Naive Bayes: fallback to variance ranking
            elif name == 'NaiveBayes':
                top_idx = np.argsort(X_train.var())[-50:]
                selected_feats = X_train.columns[top_idx].tolist()

            fs_results[name] = selected_feats
            print(f"  -> {name:12s}: {len(selected_feats):3d} features ({time.time() - start_time:.1f}s)")

        except Exception as e:
            print(f"  -> {name:12s}: ❌ Failed ({str(e)[:30]}). Using top-50 by variance.")
            fs_results[name] = X_train.columns[np.argsort(X_train.var())[-50:]].tolist()

    return fs_results

# ==================== Step 5: Main Execution Pipeline ====================
if __name__ == "__main__":
    # 1. Load and preprocess data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('ML_dataset.csv')

    if X_train is None:
        exit(1)

    # 2. Perform feature selection
    feature_selection_results = run_feature_selection(X_train, X_train_scaled, y_train)

    # 3. Generate valid model combinations (target: 113 out of 132)
    print("\n[Step 3] Generating Model Combinations...")

    classifiers = get_classifiers()
    fs_methods = list(feature_selection_results.keys())   # 12
    clf_methods = list(classifiers.keys())               # 11

    print(f"  Feature Selection Methods (FS): {len(fs_methods)}")
    print(f"  Classification Models (CLF): {len(clf_methods)}")
    print(f"  Theoretical combinations: {len(fs_methods) * len(clf_methods)} = 132")

    # Exclude redundant or identical pairs to achieve 113 unique combos
    EXCLUDED_COMBINATIONS = {
        # Self-pairs (identity redundancy)
        ('RF', 'RF'), ('GBM', 'GBM'), ('XGBoost', 'XGBoost'),
        ('NaiveBayes', 'NaiveBayes'), ('LDA', 'LDA'), ('SVM', 'SVM'),
        ('Lasso', 'Lasso'), ('Ridge', 'Ridge'),
        ('Enet', 'glmnet'),  # Both represent ElasticNet

        # Linear redundancy: Stepglm/plsRglm are linear filters
        ('Stepglm', 'Lasso'), ('Stepglm', 'Ridge'), ('Stepglm', 'glmnet'), ('Stepglm', 'LDA'),
        ('plsRglm', 'Lasso'), ('plsRglm', 'Ridge'), ('plsRglm', 'glmnet'), ('plsRglm', 'LDA'),

        # glmBoost vs tree ensembles
        ('glmBoost', 'GBM'), ('glmBoost', 'AdaBoost')
    }

    all_combinations = []
    skipped = 0

    for fs in fs_methods:
        for clf in clf_methods:
            if (fs, clf) in EXCLUDED_COMBINATIONS:
                skipped += 1
                continue
            all_combinations.append({'name': f"{fs} + {clf}", 'fs': fs, 'clf': clf})

    print(f"  Excluded redundant combinations: {skipped}")
    print(f"  Final valid combinations: {len(all_combinations)} (Target: 113)")

    # 4. Train models and evaluate via AUC
    print("\n[Step 4] Training Models and Evaluating AUC...")
    results = []
    total = len(all_combinations)

    for idx, combo in enumerate(all_combinations, 1):
        try:
            feats = feature_selection_results[combo['fs']]
            if len(feats) < 2:
                continue  # Skip if too few features

            # Use scaled data for linear models; raw for tree-based
            if combo['clf'] in ['Lasso', 'Ridge', 'glmnet', 'LDA', 'SVM']:
                X_tr, X_te = X_train_scaled[feats], X_test_scaled[feats]
            else:
                X_tr, X_te = X_train[feats], X_test[feats]

            model = clone(classifiers[combo['clf']])
            model.fit(X_tr, y_train)

            # Get prediction scores
            if hasattr(model, "predict_proba"):
                y_tr_pred = model.predict_proba(X_tr)[:, 1]
                y_te_pred = model.predict_proba(X_te)[:, 1]
            else:
                y_tr_pred = model.decision_function(X_tr)
                y_te_pred = model.decision_function(X_te)

            train_auc = roc_auc_score(y_train, y_tr_pred)
            test_auc = roc_auc_score(y_test, y_te_pred)

            # Progress logging
            if idx <= 5 or idx % 10 == 0 or idx == total:
                print(f"  [{idx:3d}/{total}] {combo['name']:35s} | Train: {train_auc:.3f} | Test: {test_auc:.3f}")

            results.append({
                'Method': combo['name'],
                'Feature_Selection': combo['fs'],
                'Classifier': combo['clf'],
                'N_Features': len(feats),
                'Train_AUC': train_auc,
                'Test_AUC': test_auc,
                'Mean_AUC': (train_auc + test_auc) / 2
            })

        except Exception as e:
            print(f"  [{idx:3d}/{total}] {combo['name']:35s} | ❌ Failed: {str(e)[:30]}")

    # 5. Save Results
    print("\n[Step 5] Saving Results...")
    if results:
        results_df = pd.DataFrame(results).sort_values('Mean_AUC', ascending=False)

        results_df.to_csv('03_all_models_results.csv', index=False)
        auc_matrix = results_df[['Method', 'Train_AUC', 'Test_AUC']].set_index('Method')
        auc_matrix.columns = ['Train', 'Test']
        auc_matrix.to_csv('04_AUC_matrix_for_R.txt', sep='\t')
        results_df.head(10).to_csv('05_top10_models.csv', index=False)

        # Save feature lists
        with open('01_feature_selection_results.txt', 'w') as f:
            for method, feats in feature_selection_results.items():
                f.write(f"{method}\t{','.join(feats)}\n")

        # Plot top 30 models
        plt.figure(figsize=(12, 10))
        top30 = results_df.head(30)
        heatmap_data = top30[['Train_AUC', 'Test_AUC']].set_index(top30['Method'])
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
        plt.title('Top 30 Model Performance (AUC)')
        plt.tight_layout()
        plt.savefig('06_python_preview_heatmap.png')
        print("  ✓ All results saved successfully.")
        print(f"  ✓ Successfully trained {len(results)} models.")
    else:
        print("❌ No models were successfully trained.")

    print("=" * 80)
    print("Pipeline Completed")
    print("=" * 80)
