# Mashkul (مَشْكُول) — Arabic Text Diacritization

Automatically add diacritics (tashkeel / حركات) to Arabic text using a
character-level BiGRU model trained on the
[arbml/tashkeela](https://huggingface.co/datasets/arbml/tashkeela) dataset.

**Input:** `كان العرب يحبون العلم ويقدرون العلماء`
**Output:** `كَانَ الْعَرَبُ يُحِبُّونَ الْعِلْمَ وَيُقَدِّرُونَ الْعُلَمَاءَ`

## Quick Start

```bash
git clone https://github.com/ahmedmgelwan/Mashkul.git
cd Mashkul

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### Get the model weights

The trained model is tiny (**~1.7 MB**) and is included in this repo under
`assets/`, so you can start using the API right away — no training needed.

It's also published on Hugging Face:
👉 **[huggingface.co/ahmedmgelwan/Mashkul](https://huggingface.co/ahmedmgelwan/Mashkul)**

```bash
# Optional — download it directly from Hugging Face instead of the repo copy
huggingface-cli download ahmedmgelwan/Mashkul --local-dir assets/
```

**Want to train it yourself instead?**
```bash
python -m src.train
```
This downloads the dataset, trains the model from scratch, and overwrites
`model.pt` and `tokenizer.json` in `assets/`.

## Try it

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/diacritize \
     -H "Content-Type: application/json" \
     -d '{"text": "كان العرب يحبون العلم ويقدرون العلماء"}'
```

Health check: `GET http://localhost:8000/health`

### Run the web demo (Streamlit)

```bash
streamlit run app.py
```
Opens a simple web UI in your browser where you can type text and see it
diacritized instantly.

## Run with Docker

```bash
docker build -t mashkul-api .
docker run --env-file .env -p 8000:8000 mashkul-api
```

## Project Structure

```
src/            Core logic: config, tokenizer, model, training, inference
api/main.py     FastAPI service (POST /diacritize)
app.py          Streamlit web demo
assets/         model.pt (~1.7 MB) + tokenizer.json — included in the repo
scripts/        Utilities: export to ONNX, upload to Hugging Face Hub
```

## Results

| Metric | Value |
|---|---|
| Character Accuracy | 96.33% |
| Word Error Rate (WER) | 13.97% |
| Diacritic Error Rate (DER) | 3.67% |
| Diacritic Error Rate — excluding last letter | 3.39% |
| Sentence Exact Match | 9.58% |
| Model size | ~1.7 MB |

## License

Apache-2.0 license