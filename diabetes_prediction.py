
#to run put python diabetes_prediction.py in terminal
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib


def main(csv_path: Path):
    # load
    df = pd.read_csv(csv_path)

    feature_cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ]
    target_col = "Outcome"

    # clean
    # Treat biologically impossible zeros as missing
    zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for c in zero_as_missing:
        df[c] = df[c].replace(0, np.nan)

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # preprocess
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocess = ColumnTransformer([("num", numeric_transformer, feature_cols)])

    # compare models
    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "LinearSVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42
        ),
    }

    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, est in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", est)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        rows.append({"Model": name, "CV ROC-AUC Mean": scores.mean(), "CV ROC-AUC Std": scores.std()})

    cv_df = pd.DataFrame(rows).sort_values("CV ROC-AUC Mean", ascending=False)
    print("\n=== 5-fold CV (ROC-AUC) ===")
    print(cv_df.to_string(index=False))

    # final train/test w best model
    best_name = cv_df.iloc[0]["Model"]
    best_est = models[best_name]
    best_pipe = Pipeline([("prep", preprocess), ("clf", best_est)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=2
    )
    best_pipe.fit(X_train, y_train)

    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_ap = average_precision_score(y_test, y_proba)
    print("\n=== Test metrics ===")
    print(f"ROC-AUC: {test_roc_auc:.3f}")
    print(f"PR AUC (Average Precision): {test_ap:.3f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))

    # save artifacts
    out = Path("diabetes_artifacts")
    out.mkdir(exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={test_roc_auc:.3f})")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.savefig(out / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(rec, prec, label=f"{best_name} (AP={test_ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.legend(loc="lower left")
    plt.savefig(out / "pr_curve.png", bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot()
    plt.title("Confusion Matrix (Test)")
    plt.savefig(out / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Save the trained pipeline
    joblib.dump(best_pipe, out / f"best_model_{best_name}.joblib")

    print(f"\nArtifacts saved to: {out.resolve()}")
    print(f"- best_model_{best_name}.joblib")
    print("- roc_curve.png, pr_curve.png, confusion_matrix.png")

    # example one off prediction
    example = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
    pred = best_pipe.predict(example)[0]
    print("\nExample prediction for (5,166,72,19,175,25.8,0.587,51):",
          "Diabetic" if pred == 1 else "Not diabetic")


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diabetes.csv")
    if not csv.exists():
        print(f"[ERROR] Could not find {csv.resolve()} — place diabetes.csv next to this script or pass a path.")
        sys.exit(1)
    main(csv)
