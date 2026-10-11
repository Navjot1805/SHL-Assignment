# Grammar Scoring Engine for Spoken Audio

Predicts a **grammar score (0-5)** from a 45-60 s speech recording (769 train / 216 test files). Evaluated with RMSE and Pearson correlation.

## Files
- `Grammar_Scoring_Lite.ipynb`: fast CPU version (about 5-10 min), acoustic and fluency features.
- `Grammar_Scoring_Engine.ipynb`: full version with Whisper transcripts, GPT-2 perplexity and embeddings (use a GPU, e.g. Colab).
- Data (add your own): `train.csv`, `test.csv`, `sample_submission.csv`, `.wav` files.

## Run
1. `pip install librosa soundfile scikit-learn scipy matplotlib seaborn tqdm joblib` (the first notebook cell does this).
2. Put the CSVs and audio in one folder and set `DATA_DIR` in the config cell.
3. **Run All**. Output: `submission.csv` (216 rows, `test.csv` order) and `submission_sample_format.csv` (ids of `sample_submission.csv`).

## Approach
Audio -> MFCC, pitch, energy and pause/fluency features (optional: tiny Whisper transcript + text features) -> Ridge, SVR and Gradient Boosting with 5-fold CV -> non-negative blend, clipped to [0, 5].

## Results
| Metric | Value |
|---|---|
| Training RMSE | 0.7741 |
| CV RMSE / Pearson | 0.7806 |
| Mean-baseline RMSE | 0.4492|


