# KNNAlgorithm

# Heart Disease Risk Screen

A small Flask web app that serves a pre-trained K-Nearest Neighbours (KNN) model. You enter 11 clinical readings, and the app reports whether the most similar past patients had heart disease.

The interface is styled like ECG graph paper. The result appears in a "monitor" panel with a heartbeat trace and a dot for each of the 5 nearest neighbours.

> **Disclaimer:** This is an educational project. It is not a medical device and does not provide a diagnosis. Talk to a doctor about any health concern.

## Project structure

```
.
├── app.py              # Flask app, form validation, prediction and HTML template
├── requirements.txt    # Python dependencies
└── knnmodel.pkl        # Trained KNeighborsClassifier (k=5)
```

All three files must be in the same folder. The HTML is embedded in `app.py`, so there is no `templates/` folder.

## The model

| Property | Value |
|---|---|
| Type | `KNeighborsClassifier` (scikit-learn) |
| Neighbours (k) | 5 |
| Training samples | 734 |
| Classes | `0` = no heart disease, `1` = heart disease |
| Scaling | None. The model expects raw, unscaled values |

### Input features

The model was trained on label-encoded data, so the app converts each dropdown choice to the integer code below.

| Feature | Meaning | Accepted values |
|---|---|---|
| `Age` | Age in years | 18 to 100 |
| `Sex` | Sex | Female = 0, Male = 1 |
| `ChestPainType` | Chest pain type | ASY (asymptomatic) = 0, ATA (atypical angina) = 1, NAP (non-anginal) = 2, TA (typical angina) = 3 |
| `RestingBP` | Resting blood pressure (mm Hg) | 50 to 250 |
| `Cholesterol` | Serum cholesterol (mg/dl) | 0 to 700 (0 = unknown) |
| `FastingBS` | Fasting blood sugar above 120 mg/dl | No = 0, Yes = 1 |
| `RestingECG` | Resting ECG result | LVH = 0, Normal = 1, ST-T abnormality = 2 |
| `MaxHR` | Maximum heart rate (beats/min) | 60 to 220 |
| `ExerciseAngina` | Chest pain during exercise | No = 0, Yes = 1 |
| `Oldpeak` | ST depression (mm) | -3 to 7 |
| `ST_Slope` | Slope of the peak exercise ST segment | Down = 0, Flat = 1, Up = 2 |

The app builds the model input in the model's own feature order, so the layout of the form can change without breaking predictions. On startup it checks that the form fields match the model's features and stops with a clear error if they do not.

## Run locally

Requires Python 3.9 to 3.12.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
python app.py
```

Open <http://localhost:5000>.

## Deploy on Render

1. Push `app.py`, `requirements.txt` and `knnmodel.pkl` to a GitHub repository.
2. On [Render](https://render.com), choose **New → Web Service** and connect the repository.
3. Use these settings:

   | Setting | Value |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn app:app` |

4. Under **Environment**, add `PYTHON_VERSION` = `3.12.3`.
5. Select **Create Web Service**. When the deploy finishes, Render gives you a public URL.

A health check is available at `/health`, which returns `{"status": "ok"}`. You can set it as the **Health Check Path** in Render.

## How a result is produced

1. The form is validated on the server (required fields, numeric ranges, allowed choices). Errors appear next to the field.
2. The values are passed to the model in the order it was trained with.
3. `predict_proba` gives the share of the 5 nearest neighbours that had heart disease.
4. The page shows the verdict and how many of the 5 similar patients had heart disease.

## Known limitations

- **No feature scaling.** KNN relies on distances, and this model was trained on unscaled data. Features with large values, such as cholesterol, have more influence than features like age. To improve this, retrain with a `StandardScaler` inside a scikit-learn `Pipeline`, pickle the whole pipeline, and replace `knnmodel.pkl`. `app.py` needs no changes.
- **Missing cholesterol.** The training data uses `0` for missing cholesterol, so entering `0` matches those records. Predictions without a real value are less reliable.
- **Small dataset.** With 734 training samples and k=5, one patient can change the result noticeably.
- **Version pinning.** `scikit-learn` is pinned to `1.6.1`, the version used to save the model. Retraining with another version means you should update the pin.

## Tech stack

- Python, Flask, Gunicorn
- scikit-learn, pandas, joblib
- Plain HTML and CSS (no JavaScript) with Bricolage Grotesque and IBM Plex Sans from Google Fonts
