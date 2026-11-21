# Healthcare Request Intelligence (HRI)

Research demonstration of a **Qwen3-0.6B** model fine-tuned with **LoRA** for administrative request categorization, priority selection, and departmental routing. **Not for clinical use or automatic patient-data disclosure.** The system is limited to 5 administrative request classes, and every prediction requires human review.

## Highlights

- 400 synthetic training examples, 100 original template-generated test examples.
- External evaluation: 68 manually labeled appointment-category requests from the Synthetic Hospital-Robot Conversational Dataset (CC BY 4.0).
- **External evaluation, 68 examples:** routing accuracy 20.59% (base) -> 88.24% (LoRA); exact match 19.12% -> 80.88%. Priority accuracy 88.24% -> 94.12%; valid JSON 100% for both.
- **Limitations:** The independent evaluation has **one category only** (`appointment`). Results do **not** demonstrate five-category generalization, clinical safety, or production readiness. Dataset and model are synthetic research artifacts.

## Repository

- `app.py`: FastAPI `GET /health` and `POST /triage` endpoints
- `common.py`: model loading and prompting aligned to the training scripts
- `train.py`, `evaluate.py`, `make_data.py`: original training/evaluation code
- `reports/`: aggregate evaluation metrics
- `.gitignore`: excludes model weights, caches, checkpoints, raw data, credentials

## Run on Colab with a GPU (private localhost demo)

1. Restore the trained LoRA adapter to `/content/hri/outputs/hri-lora` from your own Google Drive backup.
2. Upload/extract this repository to `/content/HRI_GitHub_Portfolio` (or clone it after pushing GitHub).
3. Install API dependencies in Colab (`%pip install -q fastapi uvicorn requests`). The Colab runtime should already have compatible CUDA PyTorch, Transformers and PEFT; if PEFT raises an outdated TorchAO version error, inspect and correct the package environment before starting.
4. Run from the repository directory:

   ```bash
   HRI_ADAPTER_PATH=/content/hri/outputs/hri-lora uvicorn app:app --host 127.0.0.1 --port 8000
   ```
5. In the same runtime, send POST JSON to `http://127.0.0.1:8000/triage`. See `demo_client.py`.

Do **not** expose the API with a public tunnel or accept real patient data. The API does not have authentication, rate-limiting, logging/privacy controls, or production safety validation.

## Expected JSON

```json
{"status":"model_prediction_requires_review", "result":{"category":"appointment","priority":"normal","action":"route_to_scheduling"}}
```

If output fails schema checks, it returns `{"status":"manual_review","result":null}`. This is not a clinical safety guarantee.

## Evaluation methodology

The original model and LoRA adapter were tested on identical examples. The external challenge uses manually reviewed *synthetic* hospital-robot conversation prompts, with all 68 approved examples labeled `appointment`. It measures routing behavior on that subset only. Dataset leakage/semantic overlap and generalization beyond this subset have not been comprehensively assessed. See `reports/` for the summary metrics.

## Data license and attribution

External source: [Synthetic Hospital-Robot Conversational Dataset](https://huggingface.co/datasets/Verox132/Medical_Conversational_Dataset), associated with Benedict Wohlgemuth's synthetic conversational dataset work; distributed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The HRI team manually assigned administrative routing labels to selected synthetic examples. Dataset content is not included in this source repository; the license and original source must be acknowledged when distributing adaptations.

## Resume framing

Fine-tuned an open-weight model using PEFT/LoRA and independently benchmarked the adapter on 68 manually annotated external synthetic appointment requests; routing accuracy increased from 20.59% to 88.24%, exact-match from 19.12% to 80.88%. Clearly state the single-category and synthetic-data limitations.
