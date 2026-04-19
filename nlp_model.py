"""
AgriBot NLP inference module.

Loads the pre-trained KNN + TF-IDF model saved by train_model.py
and exposes a single public function: get_chatbot_response().

Run `python train_model.py` before starting the server if the saved
artifacts do not exist yet.
"""

import json
import os

import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
ANSWERS_PATH = os.path.join(MODELS_DIR, "answers.pkl")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

# Cosine distance: 0 = identical, 1 = unrelated.
CONFIDENCE_THRESHOLD = 0.85


class AgriBotInference:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.answers = []
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """Load pre-trained model artifacts from disk."""
        missing = [
            path
            for path in [VECTORIZER_PATH, KNN_MODEL_PATH, ANSWERS_PATH]
            if not os.path.exists(path)
        ]

        if missing:
            print("\n" + "=" * 60)
            print("  [ERROR] AgriBot: trained model not found.")
            print(f"  Missing files: {[os.path.basename(path) for path in missing]}")
            print("  [ACTION] Run: python train_model.py")
            print("  Then restart the server.")
            print("=" * 60 + "\n")
            self.is_ready = False
            return

        try:
            print("[AgriBot] Loading saved model from disk...")
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.classifier = joblib.load(KNN_MODEL_PATH)
            self.answers = joblib.load(ANSWERS_PATH)
            self.is_ready = True

            if os.path.exists(META_PATH):
                with open(META_PATH, "r", encoding="utf-8") as file:
                    meta = json.load(file)
                print("[AgriBot] Model loaded successfully.")
                print(f"[AgriBot] Trained at : {meta.get('trained_at', 'N/A')}")
                print(f"[AgriBot] Samples    : {meta.get('total_samples', len(self.answers))}")
                print(
                    f"[AgriBot] Intents    : "
                    f"{len(meta.get('unique_intents', []))} categories"
                )
                print(f"[AgriBot] Vocab size : {meta.get('vectorizer_vocab', 'N/A')}")
            else:
                print(f"[AgriBot] Model loaded. ({len(self.answers)} samples ready)")

        except Exception as exc:
            print(f"[AgriBot] Failed to load model: {exc}")
            print("[AgriBot] Try running: python train_model.py")
            self.is_ready = False

    def get_response(self, text: str) -> str:
        """Return the best matching answer for the user's input text."""
        if not self.is_ready:
            return (
                "The AI model is not ready yet.\n"
                "Please run `python train_model.py` to train the model first, "
                "then restart the server."
            )

        x_test = self.vectorizer.transform([text])

        distances, indices = self.classifier.kneighbors(x_test)
        best_idx = indices[0][0]
        best_dist = distances[0][0]

        if best_dist > CONFIDENCE_THRESHOLD:
            return (
                "I'm not sure about that. I specialise in:\n\n"
                "- Crop selection by climate, weather, and soil type\n"
                "- Government schemes (PM-KISAN, PMFBY, KCC, eNAM)\n"
                "- Irrigation and water management techniques\n"
                "- Fertilizers, biofertilizers, and soil health\n"
                "- Pest and disease control\n"
                "- Seasonal and regional crop guidance\n"
                "- Organic, precision, and modern farming methods\n\n"
                "Could you rephrase or ask a more specific agriculture question?"
            )

        return self.answers[best_idx]

    def reload(self):
        """Hot-reload the model without restarting the server."""
        print("[AgriBot] Reloading model...")
        self.is_ready = False
        self._load_model()
        return self.is_ready


agri_bot = AgriBotInference()


def get_chatbot_response(user_message: str) -> str:
    """Called by main.py for every chat request."""
    return agri_bot.get_response(user_message)


def reload_model() -> bool:
    """
    Hot-reload the model after retraining.
    Returns True if model loaded successfully.
    Used by the /admin/reload-model API endpoint.
    """
    return agri_bot.reload()
