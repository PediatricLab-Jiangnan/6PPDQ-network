# ==================== Import Libraries ====================
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

# Feature Selection & Modeling Algorithm Libraries
from sklearn.linear_model import (
    Lasso, Ridge, LogisticRegression, ElasticNet
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Filter warnings to keep output clean
warnings.filterwarnings('ignore')

print("=" * 80)
print("Machine Learning Modeling Pipeline - 113 Combinations (Based on SCI papers)")
print("Supports external validation set")
print("=" * 80)

# ==================== Configuration Parameters ====================
RANDOM_SEED = 123
TRAIN_SIZE = 0.7      # Training set ratio 70%
TEST_SIZE = 0.3       # Test set ratio 30%
MAX_ITER = 5000       # Maximum iterations for linear models

# Dataset file paths
TRAIN_DATASET_PATH = 'ML_dataset.csv'           # Main dataset (for training + testing)
VALIDATION_DATASET_PATH = 'ML_dataset_validation.csv'  # External validation set

# ==================== 1. Data Loading & Preprocessing ====================
def load_raw_data(filepath, dataset_name="Main Dataset"):
    """
    Read raw dataset (without preprocessing)
    """
    print(f"\n  -> Loading {dataset_name}: {filepath}...")
    try:
        ml_data = pd.read_csv(filepath, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"  ⚠️ Warning: File not found {filepath}")
        return None, None
    
    # Handle index
    first_col = ml_data.columns[0]
    if first_col == '' or 'Unnamed' in first_col:
        ml_data = ml_data.rename(columns={first_col: 'Sample_ID'})
    
    if 'Sample_ID' in ml_data.columns:
        ml_data = ml_data.set_index('Sample_ID')

    # Check grouping column
    if 'Group' not in ml_data.columns:
        raise ValueError(f"{dataset_name} must contain a 'Group' column.")
        
    print(f"     Original dimensions: {ml_data.shape}")
    print(f"     Group distribution:")
    print(ml_data['Group'].value_counts())

    # Separate features (X) and labels (y)
    y = ml_data['Group']
    X = ml_data.drop(columns=['Group'])

    return X, y


def prepare_datasets(train_path, validation_path):
    """
    Prepare training, test, and external validation sets
    Critical fix: Align features first, then standardize
    """
    print("\n" + "="*60)
    print("[Step 1] Data Loading & Preprocessing")
    print("="*60)
    
    # Load main dataset
    X_main, y_main = load_raw_data(train_path, "Main Dataset")
    if X_main is None:
        return None, None, None, None, None, None, None, None, None, False, None
    
    # Label encoding (auto-detect)
    unique_classes = y_main.unique()
    pos_class = 'Seizures' if 'Seizures' in unique_classes else unique_classes[0]
    y_main_encoded = (y_main == pos_class).astype(int)
    print(f"\n     Label encoding: {pos_class}=1, Others=0")
    
    # Split train/test sets (70%/30%)
    print(f"\n  -> Splitting train/test sets ({int(TRAIN_SIZE*100)}%/{int(TEST_SIZE*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_main, y_main_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y_main_encoded
    )
    
    print(f"     Training set: {X_train.shape[0]} samples, {X_train.shape[1]} genes")
    print(f"     Test set: {X_test.shape[0]} samples, {X_test.shape[1]} genes")
    
    # Load external validation set (if exists)
    X_val, y_val_encoded, has_validation = None, None, False
    common_features = X_train.columns.tolist()  # Default to using all training features
    
    try:
        X_val_raw, y_val = load_raw_data(validation_path, "External Validation Set")
        if X_val_raw is not None:
            has_validation = True
            print(f"\n     External validation set original dimensions: {X_val_raw.shape}")
            print(f"     External validation set group distribution:")
            print(y_val.value_counts())
            
            # 🔥 Critical fix: Find common features first
            train_features = set(X_train.columns)
            val_features = set(X_val_raw.columns)
            common_features = sorted(list(train_features & val_features))
            
            print(f"\n     Training set gene count: {len(train_features)}")
            print(f"     Validation set gene count: {len(val_features)}")
            print(f"     Common gene count: {len(common_features)}")
            
            if len(common_features) < len(train_features):
                missing_in_val = len(train_features) - len(common_features)
                print(f"     ⚠️ Validation set is missing {missing_in_val} genes")
            
            if len(common_features) < 2:
                print(f"     ❌ Too few common genes, cannot proceed with analysis")
                has_validation = False
            else:
                # 🔥 Critical fix: Align features across all datasets before standardization
                X_train = X_train[common_features]
                X_test = X_test[common_features]
                X_val = X_val_raw[common_features]
                
                # Fill missing values in validation set (using training set mean)
                if X_val.isnull().sum().sum() > 0:
                    print(f"     ⚠️ Missing values found in validation set, filling with training set mean")
                    X_val = X_val.fillna(X_train.mean())
                
                # Validation set label encoding (using the same positive class as main dataset)
                y_val_encoded = (y_val == pos_class).astype(int)
                
                print(f"     ✓ Feature alignment complete, external validation set: {X_val.shape[0]} samples × {X_val.shape[1]} genes")
    except FileNotFoundError:
        print(f"\n  ⚠️ External validation set file not found, will use only main dataset for evaluation")
    
    # Handle missing values in training and test sets
    if X_train.isnull().sum().sum() > 0:
        print(f"\n     ⚠️ Missing values found in training set, filling with mean")
        X_train = X_train.fillna(X_train.mean())
    if X_test.isnull().sum().sum() > 0:
        X_test = X_test.fillna(X_train.mean())
    
    # 🔥 Critical fix: Standardize AFTER feature alignment
    print(f"\n  -> Standardizing data (Z-score)...")
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # Convert back to DataFrame (keep column names consistent)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=common_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=common_features, index=X_test.index)
    
    # Standardize validation set (using training set scaler)
    X_val_scaled = None
    if has_validation and X_val is not None:
        X_val_scaled_array = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled_array, columns=common_features, index=X_val.index)
        print(f"     ✓ Data standardization complete")
    else:
        print(f"     ✓ Data standardization complete (no external validation set)")
    
    return (X_train, X_test, X_val, 
            X_train_scaled, X_test_scaled, X_val_scaled, 
            y_train, y_test, y_val_encoded, has_validation, common_features)


# ==================== 2. Define Feature Selection Algorithms (12 types) ====================
def get_feature_selectors():
    """
    Return a dictionary of feature selection algorithms. 12 types in total.
    """
    return {
        'Lasso': Lasso(alpha=0.01, random_state=RANDOM_SEED, max_iter=MAX_ITER),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
        'Stepglm': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_SEED),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'RF': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'Enet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=MAX_ITER),
        'plsRglm': Ridge(alpha=0.5, random_state=RANDOM_SEED),
        'GBM': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        'NaiveBayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'glmBoost': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=RANDOM_SEED),
        'SVM': SVC(kernel='linear', random_state=RANDOM_SEED)
    }


# ==================== 3. Define Classification Modeling Algorithms (11 types) ====================
def get_classifiers():
    """
    Return a dictionary of classification modeling algorithms. 11 types in total.
    """
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


# ==================== 4. Helper Function: Execute Feature Selection ====================
def run_feature_selection(X_train, X_train_scaled, y_train):
    """
    Execute feature selection on the training set
    """
    print("\n[Step 2] Executing Feature Selection (12 methods)...")
    
    selectors = get_feature_selectors()
    fs_results = {}
    
    for name, model in selectors.items():
        start_time = time.time()
        selected_feats = []
        
        try:
            # A. Linear models (Based on Coefficients)
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

            # B. Tree model ensembles (Based on Feature Importance)
            elif name in ['XGBoost', 'RF', 'GBM', 'glmBoost']:
                X_curr = X_train
                if X_train.shape[1] > 2000:
                     vars_idx = np.argsort(X_train.var())[-2000:]
                     X_curr = X_train.iloc[:, vars_idx]
                
                model.fit(X_curr, y_train)
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-50:]
                selected_feats = X_curr.columns[top_idx].tolist()

            # C. Stepglm (L1 Logistic Regression simulation)
            elif name == 'Stepglm':
                model.fit(X_train_scaled, y_train)
                coef = np.abs(model.coef_).flatten()
                top_idx = np.argsort(coef)[-50:]
                selected_feats = X_train.columns[top_idx].tolist()

            # D. SVM / LDA (Based on Coefficients)
            elif name in ['SVM', 'LDA']:
                model.fit(X_train_scaled, y_train)
                coef = np.abs(model.coef_).flatten()
                top_idx = np.argsort(coef)[-50:]
                selected_feats = X_train.columns[top_idx].tolist()
            
            # E. NaiveBayes (Based on Variance)
            elif name == 'NaiveBayes':
                top_idx = np.argsort(X_train.var())[-50:]
                selected_feats = X_train.columns[top_idx].tolist()

            fs_results[name] = selected_feats
            print(f"  -> {name:12s}: {len(selected_feats):3d} features ({time.time()-start_time:.1f}s)")

        except Exception as e:
            print(f"  -> {name:12s}: ✗ Failed ({str(e)[:30]}). Falling back to Top 50 by variance.")
            fs_results[name] = X_train.columns[np.argsort(X_train.var())[-50:]].tolist()

    return fs_results


# ==================== 5. Main Execution Flow ====================

# 1. Load data
(X_train, X_test, X_val, 
 X_train_scaled, X_test_scaled, X_val_scaled, 
 y_train, y_test, y_val, 
 has_validation, feature_columns) = prepare_datasets(TRAIN_DATASET_PATH, VALIDATION_DATASET_PATH)

if X_train is not None:
    
    # 2. Execute feature selection (only on training set)
    feature_selection_results = run_feature_selection(X_train, X_train_scaled, y_train)

    # 3. Generate combinations
    print("\n[Step 3] Generating model combinations...")
    
    classifiers = get_classifiers()
    all_combinations = []
    
    fs_methods = list(feature_selection_results.keys())
    clf_methods = list(classifiers.keys())
    
    print(f"  Feature Selection methods (FS): {len(fs_methods)}")
    print(f"  Classification Modeling methods (CLF): {len(clf_methods)}")
    print(f"  Theoretical combinations: {len(fs_methods)} * {len(clf_methods)} = {len(fs_methods)*len(clf_methods)}")

    # Exclude duplicate combinations
    duplicates_to_skip = {
        ('RF', 'RF'), 
        ('GBM', 'GBM'), 
        ('XGBoost', 'XGBoost'),
        ('NaiveBayes', 'NaiveBayes'), 
        ('LDA', 'LDA'), 
        ('SVM', 'SVM'),
        ('Lasso', 'Lasso'), 
        ('Ridge', 'Ridge'),
        ('Enet', 'glmnet'),
        ('Stepglm', 'Lasso'),
        ('Stepglm', 'Ridge'),
        ('Stepglm', 'glmnet'),
        ('Stepglm', 'LDA'),
        ('plsRglm', 'Lasso'),
        ('plsRglm', 'Ridge'),
        ('plsRglm', 'glmnet'),
        ('plsRglm', 'LDA'),
        ('glmBoost', 'GBM'),
        ('glmBoost', 'AdaBoost')
    }

    count_generated = 0
    count_skipped = 0
    
    for fs_name in fs_methods:
        for clf_name in clf_methods:
            if (fs_name, clf_name) in duplicates_to_skip:
                count_skipped += 1
                continue
            
            all_combinations.append({
                'name': f"{fs_name} + {clf_name}",
                'fs': fs_name,
                'clf': clf_name,
            })
            count_generated += 1

    print(f"  Excluded duplicate combinations: {count_skipped}")
    print(f"  Final valid combinations: {len(all_combinations)} (Target: 113)")
    
    # 4. Train models and evaluate
    print("\n[Step 4] Training models and calculating AUC...")
    results = []
    
    total_models = len(all_combinations)
    
    for i, combo in enumerate(all_combinations, 1):
        try:
            feats = feature_selection_results[combo['fs']]
            
            if len(feats) < 2:
                continue 
            
            # Select data based on model type
            if combo['clf'] in ['Lasso', 'Ridge', 'glmnet', 'LDA', 'SVM']:
                train_data = X_train_scaled[feats]
                test_data = X_test_scaled[feats]
                val_data = X_val_scaled[feats] if (has_validation and X_val_scaled is not None) else None
            else:
                train_data = X_train[feats]
                test_data = X_test[feats]
                val_data = X_val[feats] if (has_validation and X_val is not None) else None

            model = clone(classifiers[combo['clf']])
            model.fit(train_data, y_train)
            
            # Predict
            if hasattr(model, "predict_proba"):
                y_train_pred = model.predict_proba(train_data)[:, 1]
                y_test_pred = model.predict_proba(test_data)[:, 1]
                y_val_pred = model.predict_proba(val_data)[:, 1] if (val_data is not None) else None
            else:
                y_train_pred = model.decision_function(train_data)
                y_test_pred = model.decision_function(test_data)
                y_val_pred = model.decision_function(val_data) if (val_data is not None) else None
                
            # Calculate metrics
            train_auc = roc_auc_score(y_train, y_train_pred)
            test_auc = roc_auc_score(y_test, y_test_pred)
            val_auc = roc_auc_score(y_val, y_val_pred) if (y_val_pred is not None and has_validation) else np.nan
            
            # Print progress
            if i <= 5 or i % 10 == 0 or i == total_models:
                if has_validation and not np.isnan(val_auc):
                    print(f"  [{i:3d}/{total_models}] {combo['name']:35s} | Train: {train_auc:.3f} | Test: {test_auc:.3f} | Val: {val_auc:.3f}")
                else:
                    print(f"  [{i:3d}/{total_models}] {combo['name']:35s} | Train: {train_auc:.3f} | Test: {test_auc:.3f}")
            
            results.append({
                'Method': combo['name'],
                'Feature_Selection': combo['fs'],
                'Classifier': combo['clf'],
                'N_Features': len(feats),
                'Train_AUC': train_auc,
                'Test_AUC': test_auc,
                'Validation_AUC': val_auc if has_validation else np.nan,
                'Mean_AUC': (train_auc + test_auc) / 2
            })
            
        except Exception as e:
            print(f"  [{i:3d}/{total_models}] {combo['name']:35s} | ✗ Failed: {str(e)[:30]}")

    # 5. Save results
    print("\n[Step 5] Saving results...")
    if results:
        results_df = pd.DataFrame(results).sort_values('Mean_AUC', ascending=False)
        
        # Save main CSV
        results_df.to_csv('03_all_models_results.csv', index=False)
        
        # Save AUC matrix for R (including validation set)
        if has_validation:
            auc_matrix = results_df[['Method', 'Train_AUC', 'Test_AUC', 'Validation_AUC']].set_index('Method')
            auc_matrix.columns = ['Train', 'Test', 'Validation']
        else:
            auc_matrix = results_df[['Method', 'Train_AUC', 'Test_AUC']].set_index('Method')
            auc_matrix.columns = ['Train', 'Test']
        auc_matrix.to_csv('04_AUC_matrix_for_R.txt', sep='\t')
        
        # Save Top 10
        results_df.head(10).to_csv('05_top10_models.csv', index=False)
        
        # Save feature selection list
        with open('01_feature_selection_results.txt', 'w') as f:
            for m, feats in feature_selection_results.items():
                f.write(f"{m}\t{','.join(feats)}\n")

        print("  ✓ All result files saved.")
        print(f"  ✓ Successfully trained models: {len(results)}")
        
        # 6. Plotting preview
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        top_plot = results_df.head(30)
        
        # Left plot: Train & Test AUC
        plot_data_left = top_plot[['Train_AUC', 'Test_AUC']]
        plot_data_left.index = top_plot['Method']
        sns.heatmap(plot_data_left, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=axes[0])
        axes[0].set_title('Top 30 Models Performance (Train & Test AUC)')
        
        # Right plot: If validation set exists, show Validation AUC
        if has_validation and 'Validation_AUC' in results_df.columns:
            plot_data_right = top_plot[['Test_AUC', 'Validation_AUC']]
            plot_data_right.index = top_plot['Method']
            sns.heatmap(plot_data_right, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=axes[1])
            axes[1].set_title('Top 30 Models Performance (Test & Validation AUC)')
        else:
            plot_data_right = top_plot[['Train_AUC', 'Mean_AUC']]
            plot_data_right.index = top_plot['Method']
            sns.heatmap(plot_data_right, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=axes[1])
            axes[1].set_title('Top 30 Models Performance (Train & Mean AUC)')
        
        plt.tight_layout()
        plt.savefig('06_python_preview_heatmap.png', dpi=150)
        print("  ✓ Preview heatmap saved: 06_python_preview_heatmap.png")
        
        # 7. Print best model summary
        print("\n" + "="*60)
        print("[Best Model Summary]")
        print("="*60)
        if has_validation and 'Validation_AUC' in results_df.columns:
            best_by_val = results_df.loc[results_df['Validation_AUC'].idxmax()]
            print(f"\n  Best model ranked by Validation AUC:")
            print(f"    Method: {best_by_val['Method']}")
            print(f"    Train AUC: {best_by_val['Train_AUC']:.3f}")
            print(f"    Test AUC: {best_by_val['Test_AUC']:.3f}")
            print(f"    Validation AUC: {best_by_val['Validation_AUC']:.3f}")
        
        best_by_test = results_df.loc[results_df['Test_AUC'].idxmax()]
        print(f"\n  Best model ranked by Test AUC:")
        print(f"    Method: {best_by_test['Method']}")
        print(f"    Train AUC: {best_by_test['Train_AUC']:.3f}")
        print(f"    Test AUC: {best_by_test['Test_AUC']:.3f}")
        if has_validation and not np.isnan(best_by_test['Validation_AUC']):
            print(f"    Validation AUC: {best_by_test['Validation_AUC']:.3f}")
        
    else:
        print("❌ No models trained successfully.")

print("\n" + "=" * 80)
print("Pipeline execution completed")
print("=" * 80)
