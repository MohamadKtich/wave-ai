"""
Text-based fake news detection API.

Serves the final XLM-RoBERTa classifier (trained in
notebooks/02_text_xlm_roberta_final_pipeline.ipynb) plus semantic retrieval of
similar verified news and a credibility score.

Run:
    python flask_text_pipeline.py
Then POST to http://localhost:8000/analyze_text/ with JSON: {"text": "..."}
"""
import os

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ==== Configuration ====
# Trained text classifier, hosted on the Hugging Face Hub (see models/README.md).
# Override with an env var if you fine-tune your own version.
TEXT_CLASSIFIER_MODEL = os.environ.get("WAVE_TEXT_MODEL", "Ktich/wave-xlm-roberta-fakenews")

# Embedded reference corpus used for similarity-based retrieval (real/fake news +
# their precomputed sentence embeddings). These are local data files — not part of
# this repo (see data/README.md) — point BASE_DIR at wherever you keep them.
BASE_DIR = os.environ.get("WAVE_DATA_DIR", "./data/embedded")
arabic_texts_path = os.path.join(BASE_DIR, "arabic_texts.csv")
english_texts_path = os.path.join(BASE_DIR, "english_texts.csv")
arabic_embeddings_path = os.path.join(BASE_DIR, "arabic_embeddings.pt")
english_embeddings_path = os.path.join(BASE_DIR, "english_embeddings.pt")

source_reliability = {
    "No Source": 0.1,
    "Syria TV": 0.95,
    "Al Jazeera": 0.85,
    "Free Syria News Network": 0.9,
    "Takkad": 0.95,
    "Audio Transcripts": 0.6,
    "Syriana Educational": 0.9
}

app = Flask(__name__)

# ==== Load models once at startup ====
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained(TEXT_CLASSIFIER_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(TEXT_CLASSIFIER_MODEL)
model.eval()


def detect_language(text):
    try:
        lang = detect(text)
        return 'ar' if lang == 'ar' else 'en'
    except Exception:
        return 'unknown'


def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return "Real" if predicted_class == 1 else "Fake", round(confidence * 100, 2)


def retrieve_similar_news(text, lang):
    embedding = embedder.encode(text, convert_to_tensor=True)
    if lang == 'ar':
        df = pd.read_csv(arabic_texts_path)
        embeddings = torch.load(arabic_embeddings_path)
        text_column = "Arabic Text"
    else:
        df = pd.read_csv(english_texts_path)
        embeddings = torch.load(english_embeddings_path)
        text_column = "English Text"

    cos_sim = util.cos_sim(embedding, embeddings)[0]
    top_indices = torch.topk(cos_sim, k=3).indices.tolist()

    top_results = []
    numerator, denominator = 0, 0

    for idx in top_indices:
        row = df.iloc[idx]
        sim_score = cos_sim[idx].item()
        source = row['Source']
        reliability = source_reliability.get(source, 0.4)
        label_weight = 1 if row['label'] == 1 else -1
        contribution = sim_score * reliability * label_weight
        numerator += contribution
        denominator += sim_score * reliability

        credibility = 0.6 * sim_score + 0.4 * reliability
        top_results.append({
            "text": row[text_column],
            "source": source,
            "label": "Real" if row["label"] == 1 else "Fake",
            "similarity": round(sim_score, 3),
            "credibility": round(credibility, 3)
        })

    credibility_score = max(0, min(1, (numerator / denominator if denominator != 0 else 0))) * 100
    return top_results, embedding, round(credibility_score, 2)


# ==== Flask Endpoint ====
@app.route('/analyze_text/', methods=['POST'])
def analyze_text():
    data = request.get_json()
    user_input = data.get("text")

    if not user_input:
        return jsonify({"error": "Missing input text"}), 400

    lang = detect_language(user_input)
    label, confidence = classify_news(user_input)
    similar_news, input_embedding, credibility_score = retrieve_similar_news(user_input, lang)

    return jsonify({
        "input_text": user_input,
        "language": "Arabic" if lang == "ar" else "English",
        "classification": label,
        "confidence": confidence,
        "credibility_score": credibility_score,
        "top_similar_news": similar_news
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
