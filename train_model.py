"""
AgriBot — Model Training Script
================================
Run this script once (or whenever you update your dataset) to train
the KNN model and save it to disk.

Usage:
    python train_model.py

Output:
    models/vectorizer.pkl   — trained TF-IDF vectorizer
    models/knn_model.pkl    — trained KNN classifier
    models/answers.pkl      — list of answers mapped to KNN indices
    models/questions.pkl    — list of questions (for reference)
    models/meta.json        — training metadata (date, sample count, etc.)
"""

import os
import glob
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Built-in fallback dataset (always present)
# ─────────────────────────────────────────────────────────────
BUILTIN_DATASET = [
    ("Hello", "Hello! I am AgriBot, your AI Agriculture Assistant. Ask me about crops, soil, weather, schemes, irrigation, and more!", "greeting"),
    ("Hi", "Hi there! How can I help you with your farming and agriculture needs today?", "greeting"),
    ("Hey", "Hey! Welcome to AgriBot. I can help with crops, fertilizers, government schemes, weather-based farming, and much more!", "greeting"),
    ("Who are you?", "I am AgriBot — an AI-powered agriculture assistant built using NLP and KNN. I can answer questions about farming practices, crops, soil, government schemes, weather, and much more.", "bot_identity"),
    ("What can you do?", "I can help you with: crop recommendations based on soil and climate, government agricultural schemes, irrigation techniques, pest and disease control, fertilizers, organic farming, weather-based crop selection, soil health, and post-harvest management.", "bot_identity"),
    ("Thank you", "You're welcome! Happy to help with all your agriculture needs. Feel free to ask anything else!", "farewell"),
    ("Goodbye", "Goodbye! Wishing you a great harvest season. Come back anytime!", "farewell"),
    ("What crops grow best in tropical climate?", "Tropical climates are ideal for rice, sugarcane, banana, coconut, rubber, jute, and spices like cardamom and pepper.", "climate_crop"),
    ("Which crops are suitable for arid and dry regions?", "In arid regions, grow drought-resistant crops like millets (bajra, jowar), sorghum, chickpea, cotton, groundnut, and certain wheat varieties.", "climate_crop"),
    ("What crops can be grown in cold climate?", "Cold temperate climates support wheat, barley, oats, rye, potatoes, cabbage, and carrots.", "climate_crop"),
    ("Which crops grow in monsoon climate India?", "Kharif (June-Nov): rice, maize, bajra, jowar, cotton, soybean, groundnut. Rabi (Nov-Apr): wheat, mustard, barley, gram.", "climate_crop"),
    ("What crops grow in high altitude hills?", "In high altitude regions: Apple, Pear, Plum, Peach, Walnut, and temperature crops like Wheat, Barley, Potato, and Cabbage thrive.", "climate_crop"),
    ("Which crops are best for black cotton soil?", "Black cotton soil is best for cotton, soybean, wheat, jowar, sunflower, and citrus. Its high water-retaining capacity benefits these crops.", "soil_crop"),
    ("What crops grow in red laterite soil?", "Red laterite soil is suitable for rice, ragi, groundnut, cashew, coconut, and tea.", "soil_crop"),
    ("Which crops are best for alluvial soil?", "Alluvial soil is the most fertile and supports wheat, rice, maize, sugarcane, vegetables, legumes, and oilseeds. Found mainly in Indo-Gangetic plains.", "soil_crop"),
    ("What crops suit sandy soil?", "Sandy soil is best for groundnut, potato, sweet potato, watermelon, carrot, and root vegetables.", "soil_crop"),
    ("What is kharif crop?", "Kharif crops are sown at the start of monsoon (June-July) and harvested in autumn (October). Examples: Rice, Maize, Cotton, Groundnut, Soybean, Sugarcane.", "crop_season"),
    ("What is rabi crop?", "Rabi crops are sown in winter (Oct-Nov) and harvested in spring (Mar-Apr). Examples: Wheat, Barley, Peas, Mustard, Gram.", "crop_season"),
    ("What is zaid crop?", "Zaid crops are summer crops grown between Rabi and Kharif (Mar-Jun): Watermelon, Cucumber, Bitter gourd, and Moong.", "crop_season"),
    ("What is PM-KISAN scheme?", "PM-KISAN provides direct income support of ₹6,000 per year in 3 installments to all land-holding farmers via Direct Benefit Transfer.", "govt_scheme"),
    ("What is Pradhan Mantri Fasal Bima Yojana?", "PMFBY is a crop insurance scheme with low premiums (2% Kharif, 1.5% Rabi) providing financial support against crop loss from pests, disease, and natural calamities.", "govt_scheme"),
    ("What is Soil Health Card scheme?", "The Soil Health Card Scheme provides farmers with soil nutrient status and fertilizer recommendations to improve soil health and crop productivity.", "govt_scheme"),
    ("What is Kisan Credit Card?", "KCC provides short-term credit to farmers for seeds, fertilizers, pesticides at subsidized interest rates of 4-7% per annum.", "govt_scheme"),
    ("What is eNAM?", "eNAM is a pan-India electronic agricultural market portal connecting APMC mandis to create a unified national market for commodities.", "govt_scheme"),
    ("What is MSP?", "MSP (Minimum Support Price) is the price at which the government guarantees to purchase 23 crops from farmers, protecting them from price crashes.", "govt_scheme"),
    ("What is NABARD?", "NABARD is India's apex development bank for agriculture, provides credit, refinancing, and developmental support for rural and agricultural sectors.", "govt_scheme"),
    ("What is Paramparagat Krishi Vikas Yojana?", "PKVY promotes organic farming. Farmers get ₹50,000 per hectare over 3 years for converting to organic farming practices.", "govt_scheme"),
    ("What is drip irrigation?", "Drip irrigation delivers water directly to plant roots via pipes and emitters, saving 30-50% water compared to flood irrigation.", "irrigation"),
    ("What is sprinkler irrigation?", "Sprinkler irrigation mimics rainfall by distributing water through pipes and sprinklers. Suitable for vegetables, fruits, and field crops.", "irrigation"),
    ("How much water does rice need?", "Rice requires 1,200–2,000 mm water during its growing season. Keep paddy fields flooded at 5-10 cm water depth.", "irrigation"),
    ("How much water does wheat need?", "Wheat needs 4-5 irrigations totaling 300-450 mm. Critical stages: Crown Root Initiation, tillering, flowering, and grain filling.", "irrigation"),
    ("What is organic farming?", "Organic farming uses natural manures, crop rotation, biofertilizers, and biological pest control instead of synthetic chemicals.", "farming_type"),
    ("What is hydroponics?", "Hydroponics is soil-less farming where plants grow in nutrient-rich water. It saves 95% water, allows indoor year-round cultivation.", "farming_type"),
    ("What is precision agriculture?", "Precision agriculture uses GPS, IoT sensors, and data analytics to manage crops at micro-level for maximum yield efficiency.", "farming_type"),
    ("What is NPK fertilizer?", "NPK contains Nitrogen (leaf growth), Phosphorous (root/flower development), and Potassium (plant health/disease resistance).", "fertilizer"),
    ("What is urea fertilizer?", "Urea is a nitrogen-rich fertilizer (46% N) for promoting leafy growth. Apply in 2-3 split doses during the growing season.", "fertilizer"),
    ("What is vermicompost?", "Vermicompost is produced by earthworms breaking down organic waste. Rich in nutrients and beneficial microbes. Improves soil structure.", "fertilizer"),
    ("What is biofertilizer?", "Biofertilizers are living microorganism preparations: Rhizobium (legume nitrogen fixer), Azospirillum (non-legumes), PSB (phosphate solubilizer), and Mycorrhiza.", "fertilizer"),
    ("What causes leaf curl in chili?", "Chili leaf curl is caused by Chili Leaf Curl Virus spread by whiteflies. Spray imidacloprid and remove infected plants.", "disease"),
    ("What is rice blast disease?", "Rice blast is caused by Magnaporthe oryzae. Symptoms: diamond-shaped leaf lesions. Apply tricyclazole fungicide, use resistant varieties.", "disease"),
    ("What is wheat rust?", "Wheat rust (Puccinia fungi): causes orange/yellow pustules on leaves. Use resistant varieties and apply propiconazole fungicide.", "disease"),
    ("How to treat aphids in crops?", "Spray neem oil (5 ml/L) or imidacloprid, introduce ladybirds (natural predators), and use yellow sticky traps.", "pest_control"),
    ("What is integrated pest management?", "IPM combines biological, cultural, physical, and chemical methods to control pests while minimizing harm to environment and humans.", "pest_control"),
    ("How to grow tomatoes?", "Tomatoes need well-drained loamy soil (pH 6-7), spacing of 45-60 cm, regular irrigation, staking, NPK fertilizer, and pest monitoring.", "crop_guide"),
    ("How to grow wheat?", "Sow wheat in Nov-Dec at 100-125 kg/ha. Apply DAP at sowing, urea at tillering/jointing. Irrigate at Crown Root Initiation, tillering, flowering, and grain filling.", "crop_guide"),
    ("What is millet farming?", "Millets (Bajra, Jowar, Ragi) are climate-resilient, drought-tolerant crops needing minimal inputs. Suitable for drylands. India promoted millets in 2023.", "crop_guide"),
    ("How to grow onion?", "Transplant 6-8 week old nursery seedlings at 10x7.5 cm spacing. Apply phosphorous at sowing. Stop irrigation 10 days before harvest.", "crop_guide"),
    ("What is cotton farming?", "Cotton is a Kharif crop grown in black soil. Sow in June-July at 90x45 cm spacing (Bt cotton). Manages bollworm naturally with neem spray.", "crop_guide"),
    ("What are cash crops?", "Cash crops grown for sale: Cotton, Sugarcane, Tobacco, Jute, Tea, Coffee, Rubber, and Oilseeds (Mustard, Groundnut, Sunflower).", "crop_type"),
    ("What are food crops?", "Food crops for consumption: Rice, Wheat, Maize, Barley, Millets, Pulses (Gram, Lentil, Arhar), and Vegetables.", "crop_type"),
    ("What are oilseed crops?", "Oilseed crops: Groundnut, Mustard, Sunflower, Soybean, Sesame, Linseed, Castor, and Safflower.", "crop_type"),
    ("What are spice crops?", "Major spices: Turmeric, Ginger, Chili, Black Pepper, Cardamom, Cumin, Coriander, Fenugreek, Clove, and Nutmeg.", "crop_type"),
    ("How to improve soil fertility?", "Add organic manure/compost, grow green manure crops, practice crop rotation, use balanced NPK based on soil test, and apply biofertilizers.", "soil"),
    ("What is soil pH?", "Soil pH measures acidity/alkalinity. Most crops prefer pH 6-7.5. Add lime to lower acidic pH; add gypsum/sulfur to fix alkaline soil.", "soil"),
    ("What is soil erosion?", "Soil erosion is removal of topsoil by wind/water. Prevent it by cover crops, contour farming, windbreak trees, terracing, and mulching.", "soil"),
    ("How does weather affect crops?", "Temperature affects plant metabolism, rainfall impacts moisture, humidity promotes disease, frost damages crops, and drought reduces yield. Match crops to local weather.", "weather_farming"),
    ("What crops survive drought?", "Drought-tolerant crops: Millets (Bajra, Jowar, Ragi), Moth bean, Cowpea, Cluster bean, Groundnut, Castor, and Sesame.", "weather_farming"),
    ("How to protect crops from frost?", "Irrigate before frost, cover plants with cloth, use overhead sprinklers during frost event, generate smoke in orchards, and plant frost-resistant varieties.", "weather_farming"),
    ("What is post harvest management?", "Post-harvest activities: threshing, cleaning, grading, drying, storage, packaging, and transport. Reduces losses of 20-30% and maintains quality.", "post_harvest"),
    ("How to store grains?", "Store grains in dry, rodent-proof bins. Maintain moisture below 12%. Fumigate with Aluminium Phosphide if needed.", "post_harvest"),
    ("What is Green Revolution?", "The Green Revolution (1960s-70s) introduced HYV wheat/rice, modern irrigation, and fertilizers to make India food self-sufficient. Led by Dr. M.S. Swaminathan.", "agri_general"),
    ("What is the role of technology in farming?", "Modern farming uses IoT sensors, drones, AI for yield forecasting, satellite monitoring, GPS machinery, and mobile apps for market prices.", "agri_general"),
    ("How does climate change affect agriculture?", "Climate change causes rising temperatures, erratic monsoons, extreme weather events, and shifts in pest zones, reducing crop productivity globally.", "agri_general"),
    ("What is dairy farming?", "Dairy farming involves rearing cattle (HF, Jersey, Murrah Buffalo) for milk. NDDB and cooperative societies like Amul support farmers.", "livestock"),
    ("What is aquaculture?", "Aquaculture is the controlled farming of fish, shrimp, and shellfish in ponds or tanks. PMMSY scheme supports this sector in India.", "livestock"),
]

# ─────────────────────────────────────────────────────────────
# Column aliases for flexible Excel/CSV column detection
# ─────────────────────────────────────────────────────────────
QUESTION_ALIASES = ["question", "questions", "query", "queries", "input", "text", "q", "user_input", "prompt"]
ANSWER_ALIASES   = ["answer", "answers", "response", "responses", "reply", "output", "a", "bot_response"]
INTENT_ALIASES   = ["intent", "intents", "category", "categories", "label", "labels", "tag", "class", "type"]


def _resolve_column(columns, aliases):
    col_lower = {c.strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in col_lower:
            return col_lower[alias]
    return None


def _df_to_records(df, source):
    cols  = list(df.columns)
    q_col = _resolve_column(cols, QUESTION_ALIASES)
    a_col = _resolve_column(cols, ANSWER_ALIASES)
    i_col = _resolve_column(cols, INTENT_ALIASES)

    if not q_col or not a_col:
        print(f"  ⚠️  '{source}': Cannot find question/answer columns. Available: {cols}")
        return []

    records, skipped = [], 0
    for _, row in df.iterrows():
        q = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
        a = str(row[a_col]).strip() if pd.notna(row[a_col]) else ""
        i = str(row[i_col]).strip() if (i_col and pd.notna(row[i_col])) else "general"
        if q and a and q.lower() not in ("nan", "none") and a.lower() not in ("nan", "none"):
            records.append((q, a, i))
        else:
            skipped += 1

    print(f"  ✅  '{source}': {len(records)} records loaded. ({skipped} skipped)")
    return records


def load_csv(path):
    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        return _df_to_records(df, os.path.basename(path))
    except Exception as e:
        print(f"  ⚠️  CSV error '{path}': {e}")
        return []


def load_excel(path):
    records = []
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        for sheet_name, df in sheets.items():
            df.columns = [str(c).strip() for c in df.columns]
            records.extend(_df_to_records(df, f"{os.path.basename(path)} [{sheet_name}]"))
    except Exception as e:
        print(f"  ⚠️  Excel error '{path}': {e}")
    return records


def load_all_datasets(dataset_dir):
    """Scan dataset directory and load all CSV + Excel files."""
    all_records = []
    files = (
        sorted(glob.glob(os.path.join(dataset_dir, "*.csv"))) +
        sorted(glob.glob(os.path.join(dataset_dir, "*.xlsx"))) +
        sorted(glob.glob(os.path.join(dataset_dir, "*.xls")))
    )
    if not files:
        print("  ℹ️  No external dataset files found.")
        return []

    print(f"\n  Found {len(files)} file(s): {[os.path.basename(f) for f in files]}\n")
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".csv":
            all_records.extend(load_csv(fp))
        elif ext in (".xlsx", ".xls"):
            all_records.extend(load_excel(fp))
    return all_records


# ─────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────
def train_and_save():
    print("=" * 60)
    print("  AgriBot — Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: Collect all data ─────────────────────────────
    print("\n[1/5] Loading datasets...")
    dataset = list(BUILTIN_DATASET)
    print(f"  ✅  Built-in dataset: {len(dataset)} records")

    external = load_all_datasets(DATASET_DIR)
    dataset.extend(external)

    # ── Step 2: Deduplicate ──────────────────────────────────
    print("\n[2/5] Deduplicating...")
    seen, unique = set(), []
    for q, a, i in dataset:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append((q, a, i))
    removed = len(dataset) - len(unique)
    print(f"  ✅  {len(unique)} unique samples kept. ({removed} duplicates removed)")
    dataset = unique

    # ── Step 3: Train TF-IDF + KNN ───────────────────────────
    print("\n[3/5] Training TF-IDF Vectorizer + KNN Classifier...")
    questions = [item[0] for item in dataset]
    answers   = [item[1] for item in dataset]
    intents   = [item[2] for item in dataset]
    indices   = list(range(len(questions)))

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    X = vectorizer.fit_transform(questions)

    classifier = KNeighborsClassifier(
        n_neighbors=min(5, len(questions)),  # can't exceed sample count
        metric='cosine',
        algorithm='brute',
        weights='distance',         # closer neighbours get higher voting weight
    )
    classifier.fit(X, indices)
    print(f"  ✅  KNN trained on {len(questions)} samples.")

    # ── Step 4: Quick cross-validation accuracy check ────────
    print("\n[4/5] Running cross-validation (intent labels)...")
    try:
        intent_classifier = KNeighborsClassifier(
            n_neighbors=min(3, len(questions)),
            metric='cosine',
            algorithm='brute',
        )
        cv_folds = min(5, len(questions))
        scores = cross_val_score(intent_classifier, X, intents, cv=cv_folds, scoring='accuracy')
        print(f"  📊  Cross-validation accuracy ({cv_folds}-fold): {scores.mean():.2%} ± {scores.std():.2%}")
    except Exception as e:
        print(f"  ℹ️  Cross-validation skipped: {e}")

    # ── Step 5: Save model artifacts ─────────────────────────
    print("\n[5/5] Saving model artifacts to 'models/' ...")

    joblib.dump(vectorizer,  os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(classifier,  os.path.join(MODELS_DIR, "knn_model.pkl"))
    joblib.dump(answers,     os.path.join(MODELS_DIR, "answers.pkl"))
    joblib.dump(questions,   os.path.join(MODELS_DIR, "questions.pkl"))
    joblib.dump(intents,     os.path.join(MODELS_DIR, "intents.pkl"))

    # Save human-readable metadata
    meta = {
        "trained_at":      datetime.now().isoformat(),
        "total_samples":   len(questions),
        "unique_intents":  sorted(list(set(intents))),
        "intent_counts":   {i: intents.count(i) for i in set(intents)},
        "vectorizer_vocab": len(vectorizer.vocabulary_),
        "knn_neighbors":   classifier.n_neighbors,
        "knn_metric":      classifier.metric,
        "model_files": [
            "vectorizer.pkl",
            "knn_model.pkl",
            "answers.pkl",
            "questions.pkl",
            "intents.pkl",
            "meta.json",
        ],
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  ✅  vectorizer.pkl")
    print(f"  ✅  knn_model.pkl")
    print(f"  ✅  answers.pkl")
    print(f"  ✅  questions.pkl")
    print(f"  ✅  intents.pkl")
    print(f"  ✅  meta.json")

    print("\n" + "=" * 60)
    print(f"  🎉  Training complete!")
    print(f"  📚  Samples : {len(questions)}")
    print(f"  🏷️   Intents : {len(set(intents))} unique categories")
    print(f"  📝  Vocab   : {len(vectorizer.vocabulary_)} TF-IDF features")
    print(f"  💾  Saved to: {MODELS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_save()
