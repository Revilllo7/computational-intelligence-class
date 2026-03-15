from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "iris_big.csv"
OUTPUT_DIR = BASE_DIR / "output"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

TARGET_COLUMN = "target_name"
TEST_SIZE = 0.3
RANDOM_STATE = 292583

# Task02 Decision Tree reference result (from decision_trees.py, same split)
DT_REFERENCE_ACCURACY = 0.9778
DT_REFERENCE_LABEL = "Decision Tree\n(task02)"

# Task01 Human Categorisation reference result (from human_categorisation.py, same split)
HUMAN_REFERENCE_ACCURACY = 0.98
HUMAN_REFERENCE_LABEL = "Human\n(task01)"



# Validate all required columns are present

def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



# Split dataset into train/test and separate features/target

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_set, test_set = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train = train_set[FEATURE_COLUMNS].copy()
    X_test = test_set[FEATURE_COLUMNS].copy()
    y_train = train_set[TARGET_COLUMN].copy()
    y_test = test_set[TARGET_COLUMN].copy()
    return train_set, test_set, X_train, X_test, y_train, y_test



# Confusion matrix construction and plotting

def build_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_confusion_matrix_plot(
    confusion_df: pd.DataFrame,
    classifier_name: str,
    accuracy: float,
    filepath: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt="d",
        cmap="BuGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"{classifier_name}\nAccuracy: {accuracy:.2%}", fontsize=13, pad=12)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)



# Combined accuracy bar chart for all classifiers

def save_accuracy_bar_chart(results: list[dict], filepath: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["name"] for r in results] + [DT_REFERENCE_LABEL, HUMAN_REFERENCE_LABEL]
    accuracies = [r["accuracy"] for r in results] + [DT_REFERENCE_ACCURACY, HUMAN_REFERENCE_ACCURACY]

    family_colors = {
        "knn": "lightblue",
        "naive_bayes": "lightcoral",
        "mlp": "lightgreen",
        "decision_tree": "lightyellow",
        "human": "peru",
    }

    colors = []
    for label in labels:
        if label.startswith("KNN"):
            colors.append(family_colors["knn"])
        elif label.startswith("Naive Bayes"):
            colors.append(family_colors["naive_bayes"])
        elif label.startswith("MLP"):
            colors.append(family_colors["mlp"])
        elif label.startswith("Decision Tree"):
            colors.append(family_colors["decision_tree"])
        else:
            colors.append(family_colors["human"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, [a * 100 for a in accuracies], color=colors, edgecolor="black", width=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


    # Reference lines for Decision Tree and Human accuracy

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classifier Accuracy Comparison — Iris Dataset\n(test_size=30%, random_state=292583)", fontsize=13)
    ax.axhline(y=DT_REFERENCE_ACCURACY * 100, color=family_colors["decision_tree"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(y=HUMAN_REFERENCE_ACCURACY * 100, color=family_colors["human"], linestyle="--", linewidth=1.0, alpha=0.6)



    # Legend

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=family_colors["knn"], edgecolor="black", label="KNN (k = 3, 5, 11)"),
        Patch(facecolor=family_colors["naive_bayes"], edgecolor="black", label="Naive Bayes"),
        Patch(facecolor=family_colors["mlp"], edgecolor="black", label="MLP"),
        Patch(facecolor=family_colors["decision_tree"], edgecolor="black", label="Decision Tree (task02)"),
        Patch(facecolor=family_colors["human"], edgecolor="black", label="Human categorisation (task01)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)



# Evaluate a single classifier and return metrics and confusion matrix

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, labels):
    classifier.fit(X_train, y_train)
    y_pred = pd.Series(classifier.predict(X_test), index=y_test.index, name="predicted")
    accuracy = classifier.score(X_test, y_test)
    correct = int((y_pred == y_test).sum())
    wrong = len(y_test) - correct
    confusion_df = build_confusion_matrix(y_test, y_pred, labels)
    return accuracy, correct, wrong, confusion_df



def print_classifier_block(name: str, accuracy: float, correct: int, wrong: int, total: int, confusion_df: pd.DataFrame) -> None:
    print(f"  Accuracy (score): {accuracy:.2%}")
    print(f"  Good predictions: {correct}/{total}")
    print(f"  Wrong predictions: {wrong}/{total}")
    print(f"\n")
    print("  Confusion matrix:")
    indented = confusion_df.to_string()
    for line in indented.splitlines():
        print(f"    {line}")
    print(f"\n")



def main() -> None:
    df = pd.read_csv(DATA_FILE)
    validate_schema(df)

    train_set, test_set, X_train, X_test, y_train, y_test = split_dataset(df)
    labels = sorted(df[TARGET_COLUMN].unique())
    total = len(y_test)

    classifiers = [
        ("KNN (k =3)",    KNeighborsClassifier(n_neighbors=3)),
        ("KNN (k =5)",    KNeighborsClassifier(n_neighbors=5)),
        ("KNN (k =11)",   KNeighborsClassifier(n_neighbors=11)),
        ("Naive Bayes",  GaussianNB()),
        ("MLP",          MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE)),
    ]

    print("=" * 90)
    print("MULTI-CLASSIFIER COMPARISON — IRIS DATASET")
    print("=" * 90)
    print(f"Input file: {DATA_FILE}")
    print(f"Full dataset shape: {df.shape}")
    print(f"Split configuration: train={1 - TEST_SIZE:.0%}, test={TEST_SIZE:.0%}, random_state={RANDOM_STATE}")
    print(f"Training samples: {len(X_train)}  |  Test samples: {total}")
    print(f"Classes: {labels}")
    print(f"\n")

    results = []

    for name in classifiers:
        print("-" * 90)
        print(f"  {name}")
        print("-" * 90)

        accuracy, correct, wrong, confusion_df = evaluate_classifier(
            plt.clf, X_train, y_train, X_test, y_test, labels
        )

        print_classifier_block(name, accuracy, correct, wrong, total, confusion_df)

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
        save_confusion_matrix_plot(confusion_df, name, accuracy, plot_path)
        print(f"  Confusion matrix plot saved to: {plot_path}")
        print(f"\n")

        results.append({"name": name, "accuracy": accuracy, "correct": correct, "wrong": wrong})



    # Summary table
    print("=" * 90)
    print("SUMMARY — ACCURACY COMPARISON")
    print("=" * 90)
    header = f"{'Classifier':<20} {'Accuracy':>10} {'Correct':>10} {'Wrong':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>10.2%} {r['correct']:>10}/{total} {r['wrong']:>7}/{total}")
    # Reference row
    dt_correct = round(DT_REFERENCE_ACCURACY * total)
    dt_wrong = total - dt_correct
    print(f"{'Decision Tree (task02)':<20} {DT_REFERENCE_ACCURACY:>10.2%} {dt_correct:>10}/{total} {dt_wrong:>7}/{total}  [reference]")
    human_correct = round(HUMAN_REFERENCE_ACCURACY * total)
    human_wrong = total - human_correct
    print(f"{'Human (task01)':<20} {HUMAN_REFERENCE_ACCURACY:>10.2%} {human_correct:>10}/{total} {human_wrong:>7}/{total}  [reference]")
    print(f"\n")

    # Combined accuracy bar chart
    bar_chart_path = OUTPUT_DIR / "accuracy_comparison.png"
    save_accuracy_bar_chart(results, bar_chart_path)
    print(f"Accuracy comparison chart saved to: {bar_chart_path}")
    print(f"\n")


if __name__ == "__main__":
    main()
