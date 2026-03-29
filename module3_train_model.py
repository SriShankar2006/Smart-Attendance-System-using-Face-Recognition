# ============================================================
#  MODULE 3 — MODEL TRAINING & TESTING
#  Smart Attendance System using Face Recognition
# ============================================================

import os
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


ENCODINGS_FILE = "encodings.pkl"
MODEL_FILE = "trained_model.pkl"
RESULTS_IMAGE_FILE = "training_results.png"

MIN_SAMPLES_FOR_SPLIT = 4
MIN_PER_CLASS_FOR_SPLIT = 2

CV_FOLDS_MAX = 5
CV_MIN_SAMPLES = 10


# ══════════════════════════════════════════════════════════════
# UI Helpers
# ══════════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    clear()
    print("\n======================================================")
    print(" Smart Attendance System")
    print(" MODULE 3 — Model Training & Testing")
    print("======================================================\n")


def ok(msg):
    print("✔", msg)


def info(msg):
    print("ℹ", msg)


def err(msg):
    print("✖", msg)


def section(title):
    print("\n-----", title, "-----\n")


# ══════════════════════════════════════════════════════════════
# Load Encodings
# ══════════════════════════════════════════════════════════════

def load_data():

    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)

        X = np.array(data["encodings"])
        y = np.array(data["names"])

        return X, y

    except FileNotFoundError:
        err("encodings.pkl not found — run Module 2 first.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════

def _save_charts(X_test, y_test, knn, svm, knn_acc, svm_acc):

    labels = sorted(set(y_test))

    # Predict using test data
    knn_preds = knn.predict(X_test)

    cm = confusion_matrix(y_test, knn_preds, labels=labels)

    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(1,2)

    # Confusion Matrix
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(cm, cmap="YlGn")

    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))

    ax1.set_xticklabels(labels, rotation=30)
    ax1.set_yticklabels(labels)

    ax1.set_title("KNN Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    fig.colorbar(im)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(j, i, cm[i][j],
                     ha="center",
                     va="center",
                     color="black")

    # Accuracy comparison chart
    ax2 = fig.add_subplot(gs[1])

    models = ["KNN","SVM"]
    acc = [knn_acc*100, svm_acc*100]

    bars = ax2.bar(models, acc)

    ax2.set_ylim(0,100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Model Comparison")

    for bar,value in zip(bars,acc):

        ax2.text(
            bar.get_x()+bar.get_width()/2,
            value+1,
            f"{value:.1f}%",
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(RESULTS_IMAGE_FILE)

    plt.show()

    ok("training_results.png saved")


# ══════════════════════════════════════════════════════════════
# Training Pipeline
# ══════════════════════════════════════════════════════════════

def train_and_test():

    banner()

    section("Loading Encodings")

    X, y = load_data()

    unique, counts = np.unique(y, return_counts=True)

    info(f"Total samples : {len(X)}")
    info(f"Students : {', '.join(unique)}")

    # Train/Test split
    section("Train / Test Split")

    min_count = int(counts.min())

    if len(X) < MIN_SAMPLES_FOR_SPLIT or min_count < MIN_PER_CLASS_FOR_SPLIT:

        print("Too few samples — training on all data")

        X_train = X_test = X
        y_train = y_test = y

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    info(f"Train samples : {len(X_train)}")
    info(f"Test samples : {len(X_test)}")

    # Train models
    section("Training Models")

    k = min(5, len(X_train))

    knn = KNeighborsClassifier(
        n_neighbors=k,
        algorithm="ball_tree",
        weights="distance"
    )

    knn.fit(X_train, y_train)

    svm = SVC(
        kernel="rbf",
        probability=True
    )

    svm.fit(X_train, y_train)

    # Evaluation
    section("Evaluation Results")

    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    print("KNN Accuracy :", knn_acc)
    print("SVM Accuracy :", svm_acc)

    print("\nClassification Report (KNN)\n")

    print(
        classification_report(
            y_test,
            knn.predict(X_test)
        )
    )

    # Cross validation
    if len(X) >= CV_MIN_SAMPLES:

        section("Cross Validation")

        cv_k = min(CV_FOLDS_MAX, min_count)

        scores = cross_val_score(
            knn,
            X,
            y,
            cv=cv_k
        )

        print("Scores :", scores)
        print("Mean Accuracy :", scores.mean())

    # Save charts
    section("Saving Charts")

    _save_charts(
        X_test,
        y_test,
        knn,
        svm,
        knn_acc,
        svm_acc
    )

    # Save best model
    section("Saving Model")

    best = knn if knn_acc >= svm_acc else svm

    with open(MODEL_FILE,"wb") as f:

        pickle.dump(best,f)

    ok("Best model saved → trained_model.pkl")

    print("\nNext Step → python module4_take_attendance.py\n")


# Run program

if __name__ == "__main__":

    train_and_test()