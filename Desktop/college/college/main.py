import os
import pickle

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

DATA_PATH = os.path.join(os.path.dirname(__file__), "improved_disease_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model_bundle.pkl")


def train_and_save_model(model_path=MODEL_PATH):
    data = pd.read_csv(DATA_PATH)

    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])

    X = data.drop(columns=["disease"])
    y = data["disease"]

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled,
    )

    models = {
        "rf": RandomForestClassifier(random_state=42),
        "nb": GaussianNB(),
        "svm": SVC(probability=True),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    accuracy = {
        "rf": accuracy_score(y_test, models["rf"].predict(X_test)),
        "nb": accuracy_score(y_test, models["nb"].predict(X_test)),
        "svm": accuracy_score(y_test, models["svm"].predict(X_test)),
    }

    bundle = {
        "encoder": encoder,
        "features": list(X.columns),
        "models": models,
        "accuracy": accuracy,
    }

    with open(model_path, "wb") as handle:
        pickle.dump(bundle, handle)

    print(f"Saved trained model bundle to {model_path}")
    print(f"Accuracy: {accuracy}")
    return bundle


if __name__ == "__main__":
    train_and_save_model()
