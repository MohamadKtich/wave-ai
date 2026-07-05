"""
Audio-based fake news detection API.

Pipeline: Whisper transcription -> Wav2Vec2 voice authenticity classification
(human vs AI-generated) -> text classification of the transcript -> semantic
retrieval + credibility scoring (same logic as flask_text_pipeline.py).

Run:
    python flask_api.py
Then POST to http://localhost:8000/analyze_audio/ with a multipart file field
"file", or JSON: {"url": "<google drive link>"}
"""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import re
import gc
from multiprocessing import Process, Queue

from flask import Flask, request, jsonify
import torch
import torchaudio
import torchaudio.transforms as T
import whisper
import pandas as pd
import requests
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# ==== Configuration ====
# Trained models, hosted on the Hugging Face Hub (see models/README.md).
# Override with env vars if you fine-tune your own versions.
ARABIC_VOICE_MODEL = os.environ.get("WAVE_ARABIC_AUDIO_MODEL", "Ktich/wave-wav2vec2-arabic")
ENGLISH_VOICE_MODEL = os.environ.get("WAVE_ENGLISH_AUDIO_MODEL", "Ktich/wave-wav2vec2-english")
TEXT_CLASSIFIER_MODEL = os.environ.get("WAVE_TEXT_MODEL", "Ktich/wave-xlm-roberta-fakenews")

# Embedded reference corpus for similarity-based retrieval — local data files,
# not part of this repo (see data/README.md).
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


def download_google_drive_file(url, output_path="temp_audio.wav"):
    try:
        match = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
        if not match:
            return False
        file_id = match.group(1)
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(direct_url, stream=True)
        if response.status_code != 200:
            return False
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def process_audio_job(audio_path, queue):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = whisper.load_model("large").to(device)
        result = model.transcribe(audio_path)
        transcript = result["text"].strip().lower()
        lang = result["language"]
        del model, result
        torch.cuda.empty_cache()
        gc.collect()

        meaningless = ["music", "♫", "♪", "instrumental", "melody", "background",
                       "noise", "sound effect", "sfx", "loop", "beat"]
        if len(transcript.split()) < 6 or all(w in meaningless for w in transcript.split()):
            queue.put({
                "voice_classification": None,
                "transcribed_text": None,
                "language": None,
                "news_classification": None,
                "credibility_score": None,
                "similar_news": [],
                "message": "No spoken content detected in the audio file."
            })
            return

        model_path = ARABIC_VOICE_MODEL if lang == 'ar' else ENGLISH_VOICE_MODEL
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)
        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            voice_pred = "Real" if torch.argmax(logits, dim=1).item() == 1 else "Fake"
        del model, processor, inputs, logits
        torch.cuda.empty_cache()
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(TEXT_CLASSIFIER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_CLASSIFIER_MODEL).to(device)
        inputs = tokenizer(transcript, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            news_pred = "Real" if torch.argmax(logits, dim=1).item() == 1 else "Fake"
        del model, tokenizer, inputs, logits
        torch.cuda.empty_cache()
        gc.collect()

        embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(device)
        input_embedding = embedder.encode(transcript, convert_to_tensor=True, device=device)

        if lang == "ar":
            texts_df = pd.read_csv(arabic_texts_path)
            embeddings = torch.load(arabic_embeddings_path).to(device)
            text_col = "Arabic Text"
        else:
            texts_df = pd.read_csv(english_texts_path)
            embeddings = torch.load(english_embeddings_path).to(device)
            text_col = "English Text"

        cos_sim = util.cos_sim(input_embedding, embeddings)[0]
        top_indices = torch.topk(cos_sim, k=3).indices.tolist()

        results, num, denom = [], 0, 0
        for idx in top_indices:
            row = texts_df.iloc[idx]
            sim = cos_sim[idx].item()
            source = row["Source"]
            rel = source_reliability.get(source, 0.4)
            label_weight = 1 if row["label"] == 1 else -1
            num += sim * rel * label_weight
            denom += sim * rel
            results.append({
                "text": row[text_col],
                "source": source,
                "label": "Real" if row["label"] == 1 else "Fake",
                "similarity": round(sim, 3),
                "reliability": rel
            })

        score = max(0, min(1, num / denom)) * 100 if denom != 0 else 0
        torch.cuda.empty_cache()
        gc.collect()

        queue.put({
            "voice_classification": voice_pred,
            "transcribed_text": transcript,
            "language": "Arabic" if lang == "ar" else "English",
            "news_classification": news_pred,
            "credibility_score": round(score, 2),
            "similar_news": results
        })
    except Exception as e:
        queue.put({"error": str(e)})
        torch.cuda.empty_cache()
        gc.collect()


# ==== Endpoint ====
@app.route('/analyze_audio/', methods=['POST'])
def analyze_audio():
    temp_path = "temp_audio.wav"

    if 'file' in request.files:
        file = request.files['file']
        file.save(temp_path)
    else:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing file or URL"}), 400
        if not download_google_drive_file(data['url'], temp_path):
            return jsonify({"error": "Failed to download file from Google Drive"}), 400

    queue = Queue()
    p = Process(target=process_audio_job, args=(temp_path, queue))
    p.start()
    p.join()
    result = queue.get()
    os.remove(temp_path)
    torch.cuda.empty_cache()
    gc.collect()
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
