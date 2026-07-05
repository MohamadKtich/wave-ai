# Dataset

## Why there's no data in this repository

The full datasets used to train the models are **not published here**:

- **Full text dataset**: collected Dec 2024 – Jun 2025 from multiple news outlets and
  channels, including sources covering sensitive, politically-charged topics (particularly
  Syria-related news). Publishing it in full raises privacy/political-sensitivity concerns
  beyond the scope of a public portfolio repo.
- **Audio dataset**: contains both real recorded speech and AI-generated (TTS) audio
  clips. Publishing synthetic voice clips that could be mistaken for real people's speech,
  or real recordings without clear redistribution rights, is avoided here.

## Requesting access

The full datasets are available for academic/research purposes on request. Please open
an issue in this repository or reach out via [Ktichmohamad@gmail.com](mailto:Ktichmohamad@gmail.com) describing
your intended use.

## Folder expected by the API (`WAVE_DATA_DIR`)

`src/api/flask_text_pipeline.py` and `src/api/flask_api.py` expect a local folder
(default `./data/embedded`, override with the `WAVE_DATA_DIR` env var) containing:

```
embedded/
├── arabic_texts.csv        # columns: Arabic Text, Source, label
├── english_texts.csv       # columns: English Text, Source, label
├── arabic_embeddings.pt    # SentenceTransformer embeddings, same row order as arabic_texts.csv
└── english_embeddings.pt   # SentenceTransformer embeddings, same row order as english_texts.csv
```

These are generated in `02_text_xlm_roberta_final_pipeline.ipynb` — request the full
dataset (see above) or regenerate them from your own labeled corpus using the same
`SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` model.

## Reproducing the dataset yourself

The construction methodology is fully documented in the code:
- Text: see `01_text_preprocessing_and_baselines.ipynb` and `02_text_xlm_roberta_final_pipeline.ipynb`
  for cleaning and labeling logic.
- Audio: see `05_audio_wav2vec2_finetuning.ipynb` for segmenting/standardizing clips to
  5-second windows and the list of TTS engines used to generate synthetic samples
  (VITS, Piper, OpenVoice, Bark, Coqui, etc.).

You can collect your own dataset following the same structure:

```
text: id, text, label (real/fake), source, language
audio: file_path, label (real/fake), source/tts_engine, duration_sec
```

