"""
AgriBot — NLP Inference Module
================================
Loads the pre-trained KNN + TF-IDF model saved by train_model.py
and exposes a single public function: get_chatbot_response()

IMPORTANT:
    Run  `python train_model.py`  before starting the server.
    If no saved model is found, a clear error message is logged.
"""

import os
import json
import joblib

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
KNN_MODEL_PATH  = os.path.join(MODELS_DIR, "knn_model.pkl")
ANSWERS_PATH    = os.path.join(MODELS_DIR, "answers.pkl")
META_PATH       = os.path.join(MODELS_DIR, "meta.json")

# Low-confidence threshold (cosine distance: 0 = identical, 1 = unrelated)
CONFIDENCE_THRESHOLD = 0.85

# ─────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────
class AgriBotInference:
    def __init__(self):
        self.vectorizer  = None
        self.classifier  = None
        self.answers     = []
        self.is_ready    = False
        self._load_model()

    def _load_model(self):
        """Load pre-trained model artifacts from disk."""
        missing = [
            p for p in [VECTORIZER_PATH, KNN_MODEL_PATH, ANSWERS_PATH]
            if not os.path.exists(p)
        ]

        if missing:
            print("\n" + "=" * 60)
            print("  ❌  AgriBot: Trained model NOT found!")
            print(f"     Missing files: {[os.path.basename(m) for m in missing]}")
            print("  👉  Run:  python train_model.py")
            print("     Then restart the server.")
            print("=" * 60 + "\n")
            self.is_ready = False
            return

        try:
            print("[AgriBot] 🔄 Loading saved model from disk...")
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.classifier = joblib.load(KNN_MODEL_PATH)
            self.answers    = joblib.load(ANSWERS_PATH)
            self.is_ready   = True

            # Print metadata summary if available
            if os.path.exists(META_PATH):
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                print(f"[AgriBot] ✅ Model loaded successfully!")
                print(f"[AgriBot] 📅 Trained at : {meta.get('trained_at', 'N/A')}")
                print(f"[AgriBot] 📚 Samples    : {meta.get('total_samples', len(self.answers))}")
                print(f"[AgriBot] 🏷️  Intents    : {len(meta.get('unique_intents', []))} categories")
                print(f"[AgriBot] 📝 Vocab size : {meta.get('vectorizer_vocab', 'N/A')}")
            else:
                print(f"[AgriBot] ✅ Model loaded. ({len(self.answers)} samples ready)")

        except Exception as e:
            print(f"[AgriBot] ❌ Failed to load model: {e}")
            print("[AgriBot] 👉 Try running:  python train_model.py")
            self.is_ready = False

    def get_response(self, text: str) -> str:
        """Return the best matching answer for the user's input text."""

        if not self.is_ready:
            return (
                "⚠️ The AI model is not ready yet.\n"
                "Please run `python train_model.py` to train the model first, "
                "then restart the server."
            )

        # Vectorize the query
        X_test = self.vectorizer.transform([text])

        # KNN lookup
        distances, indices = self.classifier.kneighbors(X_test)
        best_idx  = indices[0][0]
        best_dist = distances[0][0]

        # Reject low-confidence matches
        if best_dist > CONFIDENCE_THRESHOLD:
            return (
                "I'm not sure about that. I specialise in:\n\n"
                "🌦️ Crop selection by climate, weather & soil type\n"
                "🏛️ Government schemes (PM-KISAN, PMFBY, KCC, eNAM...)\n"
                "💧 Irrigation & water management techniques\n"
                "🧪 Fertilizers, biofertilizers & soil health\n"
                "🦠 Pest & disease control\n"
                "📅 Seasonal & regional crop guidance\n"
                "🌿 Organic, precision & modern farming methods\n\n"
                "Could you rephrase or ask a more specific agriculture question?"
            )

        return self.answers[best_idx]

    def reload(self):
        """Hot-reload the model without restarting the server."""
        print("[AgriBot] 🔄 Reloading model...")
        self.is_ready = False
        self._load_model()
        return self.is_ready


# ─────────────────────────────────────────────────────────────
# Singleton instance — loaded once when the server starts
# ─────────────────────────────────────────────────────────────
agri_bot = AgriBotInference()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
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
