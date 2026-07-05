# Trained Models

Model checkpoints are **not stored in this repository** (each is 500MB–2GB, which is
impractical for a Git repo). Instead, they're hosted on the **Hugging Face Hub** and
loaded directly at inference time.

## Models & links

| Model | Task | Base architecture | Hugging Face Hub link |
|---|---|---|---|
| Text fake-news classifier | Binary classification (real / fake), Arabic + English | `xlm-roberta-large` | [Ktich/wave-xlm-roberta-fakenews](https://huggingface.co/Ktich/wave-xlm-roberta-fakenews) |
| Audio authenticity classifier (Arabic) | Binary classification (human / AI-generated voice) | `wav2vec2-large-xlsr-53` (Arabic) | [Ktich/wave-wav2vec2-arabic](https://huggingface.co/Ktich/wave-wav2vec2-arabic) |
| Audio authenticity classifier (English) | Binary classification (human / AI-generated voice) | `wav2vec2-large-xlsr-53` (English) | [Ktich/wave-wav2vec2-english](https://huggingface.co/Ktich/wave-wav2vec2-english) |

## How to upload your trained models

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF access token (from huggingface.co/settings/tokens)
```

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/wave-xlm-roberta-fakenews", exist_ok=True)

# if you saved the model locally with model.save_pretrained("path/to/model")
api.upload_folder(
    folder_path="path/to/your/saved_model",
    repo_id="your-username/wave-xlm-roberta-fakenews",
)
```

Repeat for the two Wav2Vec2 models.

## How to load a model for inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Ktich/wave-xlm-roberta-fakenews")
model = AutoModelForSequenceClassification.from_pretrained("Ktich/wave-xlm-roberta-fakenews")
```

```python
from transformers import AutoProcessor, AutoModelForAudioClassification

# Arabic
processor_ar = AutoProcessor.from_pretrained("Ktich/wave-wav2vec2-arabic")
model_ar = AutoModelForAudioClassification.from_pretrained("Ktich/wave-wav2vec2-arabic")

# English
processor_en = AutoProcessor.from_pretrained("Ktich/wave-wav2vec2-english")
model_en = AutoModelForAudioClassification.from_pretrained("Ktich/wave-wav2vec2-english")
```

The deployment notebooks (`03_text_api_deployment.ipynb`, `06_audio_api_deployment.ipynb`)
expect models to be loadable this way once you update the repo IDs above.
