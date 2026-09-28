# Publishing this project on GitHub

## 1. Get the code onto GitHub

### Option A — command line (from inside this project folder)

```bash
git init
git add .
git commit -m "House prices: regression, regularization, bias-variance analysis"
```

Create the empty repo on github.com first (**New repository** → name it
`house-price-prediction` → skip the README), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-prediction.git
git push -u origin main
```

### Option B — GitHub Desktop (no terminal)

1. **File → Add local repository** → select this folder
   (it will say "not a repository" → click **Create a repository**).
2. Confirm the suggested `.gitignore` rules, then **Publish repository**.

## 2. What gets committed (and what doesn't)

The shipped `.gitignore` excludes `.venv/`, `__pycache__/`, `*.csv`,
`artifacts/*.joblib` and logs. `figures/` and `artifacts/metrics.json` are
committed so recruiters can see results without training anything.

No dataset handling is needed at all: the California Housing dataset ships
with scikit-learn and is cached automatically on first run of
`python train_house.py`.

## 3. Optional: Hugging Face Space demo

The trained forest is ~90 MB — too large to upload comfortably. The clean
pattern is to let the Space train on startup:

1. https://huggingface.co → **New Space** → SDK **Gradio**.
2. Upload `app.py`, `transformers.py`, `train_house.py`,
   `requirements.txt`, and a `README.md` with the header:

```yaml
---
title: California House Price Predictor
emoji: 🏠
sdk: gradio
app_file: app.py
---
```

3. Add `import subprocess; subprocess.run(["python", "train_house.py"])`
   at the top of `app.py` (free CPUs finish it in a few minutes), and push.
4. Link the live Space URL at the top of your GitHub README.

## 4. Housekeeping

- **Description:** `House price regression on California Housing (20,640
  block groups) — Linear/Ridge/Lasso/RF comparison, learning-curve
  bias-variance analysis, RMSE down 40%+ vs baseline, Gradio demo.`
  *(check `artifacts/metrics.json` for the exact final figure)*
- **Topics:** `machine-learning`, `regression`, `scikit-learn`,
  `bias-variance`, `gradio`, `data-science-portfolio`
- Add `figures/learning_curves.png` and `figures/pred_vs_actual.png` to the
  README — they are the visual proof of the bias-variance work.
