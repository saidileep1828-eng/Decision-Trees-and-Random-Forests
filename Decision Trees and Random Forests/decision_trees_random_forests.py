import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(path="heart.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Please place the heart disease CSV in the working directory."
        )
    df = pd.read_csv(path)
    return df


def main():
    df = load_dataset("heart.csv")

    print("Dataset head:\n", df.head(), "\n")
    print("Dataset info:\n")
    print(df.info(), "\n")
    print("Dataset description:\n", df.describe(), "\n")

    if "target" not in df.columns:
        raise ValueError("Expected column 'target' not found in dataset.")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
    print("Decision Tree Training Accuracy:", dt.score(X_train, y_train))
    print("Decision Tree Testing Accuracy:", dt.score(X_test, y_test))

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=X.columns,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig("decision_tree.png")
    plt.close()
    print("Saved decision tree visualization: decision_tree.png")

    dt_depth = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_depth.fit(X_train, y_train)
    y_pred_dt_depth = dt_depth.predict(X_test)
    print("Controlled Tree Accuracy (max_depth=4):", accuracy_score(y_test, y_pred_dt_depth))
    print(
        "Controlled Tree Classification Report:\n",
        classification_report(y_test, y_pred_dt_depth),
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

    print("Comparison:")
    print("  Decision Tree:", accuracy_score(y_test, y_pred_dt))
    print("  Random Forest:", accuracy_score(y_test, y_pred_rf))

    scores = cross_val_score(rf, X, y, cv=5, n_jobs=1)
    print("Cross Validation Scores:", scores)
    print("Average Accuracy:", scores.mean())

    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    print("Feature Importances:\n", importance)

    plt.figure(figsize=(10, 8))
    plt.barh(importance["Feature"], importance["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature importance plot: feature_importance.png")


if __name__ == "__main__":
    main()
