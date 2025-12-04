import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_and_save_model():
    # 1. Load data
    iris = load_iris()
    X = iris.data  # features
    y = iris.target  # labels

    # 2. Train-test split (just for practice, not strictly needed to save model)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Create model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4. Train model
    clf.fit(X_train, y_train)

    # 5. Print accuracy for info
    acc = clf.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2f}")

    # 6. Save model to file
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Model saved to model.pkl")


if __name__ == "__main__":
    train_and_save_model()
