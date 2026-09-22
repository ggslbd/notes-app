import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "note_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

DEFAULT_CATEGORIES = ["personal", "work", "health", "shopping", "idea", "other"]

class NoteClassifier:
    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.categories: list[str] = list(DEFAULT_CATEGORIES)
        self._load()

    def _load(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            try:
                self.pipeline = joblib.load(MODEL_PATH)
                import json
                with open(LABEL_ENCODER_PATH) as f:
                    data = json.load(f)
                    self.categories = data.get("categories", list(DEFAULT_CATEGORIES))
            except Exception:
                self.pipeline = None

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.pipeline is not None:
            joblib.dump(self.pipeline, MODEL_PATH)
            import json
            with open(LABEL_ENCODER_PATH, "w") as f:
                json.dump({"categories": self.categories}, f)

    def train(self, texts: list[str], labels: list[str]):
        """Train or re-train the classifier on (text, label) pairs."""
        # Build category list: existing + any new labels
        for lbl in labels:
            if lbl not in self.categories:
                self.categories.append(lbl)
        y = np.array([self.categories.index(lbl) for lbl in labels])

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])
        self.pipeline.fit(texts, y)
        self.save()

    def predict(self, text: str) -> dict:
        """Classify a single note. Returns category + confidence."""
        if self.pipeline is None or len(self.categories) == 0:
            return {"category": "other", "confidence": 0.0, "trained": False}

        probs = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        category = self.categories[idx]
        confidence = float(probs[idx])
        return {"category": category, "confidence": round(confidence, 3), "trained": True}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def is_trained(self) -> bool:
        return self.pipeline is not None


_classifier = NoteClassifier()

def train(texts: list[str], labels: list[str]):
    _classifier.train(texts, labels)

def predict(text: str) -> dict:
    return _classifier.predict(text)

def get_categories() -> list[str]:
    return _classifier.categories

def trained_status() -> bool:
    return _classifier.is_trained()
