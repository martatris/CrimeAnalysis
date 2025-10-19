import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.utils import class_weight
import joblib

# ---- Config ----
DATA_PATH = "crime_data.csv"
TARGET = "Victim_Fatal_Status"   # target variable (binary: e.g., 'Yes'/'No' or 1/0)
MODEL_OUTPUT_DIR = "crime_model_output"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ---- Helper functions ----
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df

def preview(df, n=5):
    print("Columns:", df.columns.tolist())
    print(df.head(n))        # print head instead of display
    print("\nInfo:")
    print(df.info())

def basic_cleaning(df):
    # Standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Trim string columns
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # Normalize common target values if present
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace({
            "YES":"Yes","NO":"No",
            "Y":"Yes","N":"No",
            1:"Yes", 0:"No"
        }).astype('category')

    # Convert ages to numeric (coerce errors)
    for age_col in ["Offender_Age", "Victim_Age"]:
        if age_col in df.columns:
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')

    return df

def eda(df, save_plots=False):
    print("=== EDA SUMMARY ===")
    # 1) Basic target distribution
    if TARGET in df.columns:
        print("\nTarget distribution:")
        display(df[TARGET].value_counts(dropna=False))
        sns.countplot(y=TARGET, data=df, order=df[TARGET].value_counts().index)
        plt.title(f"Distribution of {TARGET}")
        plt.tight_layout()
        if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "target_distribution.png"))
        plt.show()

    # 2) Category counts
    if "Category" in df.columns:
        plt.figure(figsize=(8,6))
        topn = 15
        sns.countplot(y="Category", data=df, order=df['Category'].value_counts().index[:topn])
        plt.title(f"Top {topn} Categories")
        plt.tight_layout()
        if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "top_categories.png"))
        plt.show()

    # 3) Report Type counts
    if "Report Type" in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(y="Report Type", data=df, order=df['Report Type'].value_counts().index)
        plt.title("Report Type distribution")
        plt.tight_layout()
        if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "report_type.png"))
        plt.show()

    # 4) Age distributions
    for age_col in ["Offender_Age", "Victim_Age"]:
        if age_col in df.columns:
            plt.figure(figsize=(8,4))
            sns.histplot(df[age_col].dropna(), bins=30, kde=False)
            plt.title(f"{age_col} distribution (non-null)")
            plt.xlabel("Age")
            plt.tight_layout()
            if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"{age_col}_hist.png"))
            plt.show()

    # 5) Cross-tab: Offender_Race vs Victim_Race
    if "Offender_Race" in df.columns and "Victim_Race" in df.columns:
        ct = pd.crosstab(df['Offender_Race'], df['Victim_Race'])
        print("\nCross-tab Offender_Race vs Victim_Race (top rows):")
        display(ct.head())
        plt.figure(figsize=(10,6))
        sns.heatmap(ct, cmap="Blues", annot=False)
        plt.title("Offender_Race vs Victim_Race")
        plt.tight_layout()
        if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "race_crosstab_heatmap.png"))
        plt.show()

    # 6) Offender_Gender vs Victim_Gender counts
    if "Offender_Gender" in df.columns and "Victim_Gender" in df.columns:
        plt.figure(figsize=(6,4))
        cross = pd.crosstab(df['Offender_Gender'], df['Victim_Gender'])
        sns.heatmap(cross, annot=True, fmt="d")
        plt.title("Offender_Gender vs Victim_Gender")
        plt.tight_layout()
        if save_plots: plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "gender_crosstab.png"))
        plt.show()

    # 7) Missingness overview
    miss = df.isna().mean().sort_values(ascending=False)
    print("\nMissingness (fraction):")
    display(miss[miss>0].head(20))

def prepare_features(df, target=TARGET):
    """
    Prepare feature matrix X and target y.
    We will:
      - Keep a subset of useful columns
      - Impute missing ages, drop rows with missing target
      - Encode categorical features via pipeline
    """
    required_cols = [
        "OffenderStatus", "Offender_Race", "Offender_Gender", "Offender_Age",
        "PersonType", "Victim_Race", "Victim_Gender", "Victim_Age",
        "Report Type", "Category", "Disposition"
    ]
    # Keep only available columns from required list
    features = [c for c in required_cols if c in df.columns]
    print("Using features:", features)

    # Drop rows missing target
    if target in df.columns:
        df = df[~df[target].isna()].copy()
        y = df[target].copy()
    else:
        raise KeyError(f"Target column {target} not found in data")

    X = df[features].copy()
    return X, y

def build_model_pipeline(X):
    # Identify numeric and categorical columns
    numeric_features = [c for c in X.columns if "Age" in c]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer - use onehot (safe) with handle_unknown
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Estimator
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
    return pipeline, numeric_features, categorical_features

def train_and_evaluate(pipeline, X, y, test_size=0.2):
    # Map target to binary labels 0/1
    # If target is categorical like 'Yes'/'No', map appropriately
    y_clean = y.copy()
    if y_clean.dtype.name == 'category' or y_clean.dtype == object:
        y_clean = y_clean.astype(str)
    # create binary 0/1 if it looks like Yes/No
    if set(y_clean.dropna().unique()).issubset(set(['Yes','No','yes','no','Y','N','y','n','1','0','True','False','true','false'])):
        y_bin = y_clean.str.lower().map(lambda v: 1 if str(v).lower() in ['yes','y','1','true'] else 0)
    else:
        # attempt to convert numeric categories (if already 0/1)
        try:
            y_bin = pd.to_numeric(y_clean).fillna(0).astype(int)
        except:
            # fallback: label encode
            y_bin = pd.factorize(y_clean)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=test_size, random_state=42, stratify=y_bin)

    # Fit
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None

    # Metrics
    print("\n=== Evaluation on test set ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print("ROC AUC:", auc)
        except:
            pass
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
        plt.plot([0,1],[0,1],'--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return pipeline, X_train, X_test, y_train, y_test, y_pred, y_proba

def feature_importances(pipeline, numeric_features, categorical_features, top_n=25):
    """
    Extract feature names after OneHot encoding and show importances for RandomForest
    """
    clf = pipeline.named_steps['clf']
    preproc = pipeline.named_steps['preprocessor']
    # get cat feature names from onehot
    cat_pipe = preproc.named_transformers_['cat']
    ohe = cat_pipe.named_steps['onehot']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    feat_names = numeric_features + cat_feature_names

    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("\nTop feature importances:")
    display(fi.head(top_n))
    plt.figure(figsize=(8, min(0.25*top_n,8)))
    fi.head(top_n).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Feature importances (top {})".format(top_n))
    plt.tight_layout()
    plt.show()
    return fi

def save_artifacts(pipeline, fi, out_dir=MODEL_OUTPUT_DIR):
    joblib.dump(pipeline, os.path.join(out_dir, "crime_victim_fatal_model.pkl"))
    fi.to_csv(os.path.join(out_dir, "feature_importances.csv"))
    print(f"Saved pipeline to {os.path.join(out_dir, 'crime_victim_fatal_model.pkl')}")
    print(f"Saved feature importances to {os.path.join(out_dir, 'feature_importances.csv')}")

# ---- Main script execution ----
if __name__ == "__main__":
    # 1) Load
    df = load_data(DATA_PATH)

    # 2) Preview
    print("\nPreview and schema:")
    preview(df, n=3)

    # 3) Clean
    df = basic_cleaning(df)

    # 4) EDA
    eda(df, save_plots=True)

    # 5) Prepare features + target
    X, y = prepare_features(df, target=TARGET)

    # 6) Build pipeline
    pipeline, num_feats, cat_feats = build_model_pipeline(X)

    # 7) Train & evaluate
    pipeline_trained, X_train, X_test, y_train, y_test, y_pred, y_proba = train_and_evaluate(pipeline, X, y, test_size=0.2)

    # 8) Feature importances
    fi = feature_importances(pipeline_trained, num_feats, cat_feats, top_n=25)

    # 9) Save artifacts
    save_artifacts(pipeline_trained, fi)

    print("\n=== Done ===")
    print(f"Look in {MODEL_OUTPUT_DIR} for saved model & artifacts.")

