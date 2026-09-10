# CaO Calcium-Looping TGA Kinetics Analyzer

A Streamlit app for analyzing thermogravimetric (TGA) data from CaO
calcium-looping cycles.

## What it does

- Upload one or more TGA CSV exports (Time / Weight / Program Temperature columns).
- Drag-select each cycle's carbonation window directly on an interactive chart
  (an optional temperature-plateau auto-detector can also propose candidate windows).
- Computes the CO2 capture capacity (mmol CO2 / g sorbent) for every defined cycle.
- Fits 11 standard calcium-looping kinetic models to each cycle's conversion
  curve, each as a **two-stage fit**: a fast reaction-controlled stage and a
  slow diffusion-controlled stage, with the transition point between them
  independently optimized *per model*.
- Reports R2, AIC, and BIC for every model so you can compare and pick the best one.
- Lets you mix a different model for the fast stage vs. the slow stage.
- Compares capture capacity and conversion curves across cycles, including an
  optional Grasa-Abanades capacity-decay fit across cycle number.
- Exports every table as CSV/Excel and every figure as PNG/PDF/SVG
  (publication-quality, 300 dpi / vector).

## Files

```
app.py               Streamlit app (entry point)
kinetic_models.py     Kinetic model equations, two-stage fitting engine, AIC/BIC
data_utils.py          CSV parsing, cycle extraction, capacity calculations, export helpers
figures.py             Matplotlib figure builders (publication-style)
requirements.txt       Python dependencies
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Put it on GitHub

```bash
cd cao_kinetics_app
git init
git add app.py kinetic_models.py data_utils.py figures.py requirements.txt README.md .gitignore
git commit -m "Initial CaO kinetics app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **New app**, pick the repository and branch you just pushed.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**. The first build installs `requirements.txt` and typically
   takes a minute or two; the app then gets a persistent public/private URL.
5. To update the app later, just push new commits to the same branch —
   Streamlit Cloud redeploys automatically.

## Data format expected

A standard TA Instruments-style TGA export CSV with (at minimum) these columns
(names are matched case-insensitively and don't need to match exactly):

- `Time` (minutes)
- `Weight` / `Unsubtracted Weight` (mg by default — switch to µg in the sidebar
  if your export is in micrograms)
- `Program Temperature` (°C) — used for the optional auto-detect helper and for
  the temperature trace on the selection chart; not strictly required otherwise.

## Notes on the calculations

- **Capacity (mmol CO2/g)** is computed directly from the mass gain during the
  selected carbonation window, on a whole-sorbent basis: it does not depend on
  the "active CaO fraction" setting.
- **Fractional conversion X(t)**, used only for kinetic model fitting, is
  normalized by the theoretical maximum mass gain of the sorbent's active CaO
  content (set the "Active CaO mass fraction" in the sidebar to less than 1.0
  for supported/composite sorbents, e.g. Ni/CaO dual-functional materials).
- See the in-app **About / model reference** tab for the full list of kinetic
  models, their equations, and the AIC/BIC formulas used to compare them.
