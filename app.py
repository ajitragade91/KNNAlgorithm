"""
Heart disease risk screen - Flask app for a KNN classifier (knnmodel.pkl).

The model was trained on label-encoded, unscaled data, so every categorical
answer below is mapped to the same integer code used during training:

    Sex             F=0, M=1
    ChestPainType   ASY=0, ATA=1, NAP=2, TA=3
    FastingBS       0 / 1
    RestingECG      LVH=0, Normal=1, ST=2
    ExerciseAngina  N=0, Y=1
    ST_Slope        Down=0, Flat=1, Up=2

Run locally:   python app.py
Run on Render: gunicorn app:app

Files needed: app.py, requirements.txt, knnmodel.pkl (same folder).
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knnmodel.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# --------------------------------------------------------------------------
# Form definition. Layout order is free; FEATURES (below) sets the model order.
# --------------------------------------------------------------------------
GROUPS = [
    {
        "title": "Background",
        "fields": [
            {"name": "Age", "label": "Age", "kind": "number", "unit": "years",
             "min": 18, "max": 100, "step": 1, "placeholder": "e.g. 54"},
            {"name": "Sex", "label": "Sex", "kind": "select",
             "options": [(0, "Female"), (1, "Male")]},
            {"name": "ChestPainType", "label": "Chest pain type", "kind": "select",
             "wide": True,
             "options": [(0, "Asymptomatic (no chest pain)"),
                         (1, "Atypical angina"),
                         (2, "Non-anginal pain"),
                         (3, "Typical angina")]},
        ],
    },
    {
        "title": "At rest",
        "fields": [
            {"name": "RestingBP", "label": "Resting blood pressure", "kind": "number",
             "unit": "mm Hg", "min": 50, "max": 250, "step": 1, "placeholder": "e.g. 130"},
            {"name": "Cholesterol", "label": "Serum cholesterol", "kind": "number",
             "unit": "mg/dl", "min": 0, "max": 700, "step": 1, "placeholder": "e.g. 220",
             "hint": "Enter 0 if you don't have this value."},
            {"name": "FastingBS", "label": "Fasting blood sugar above 120 mg/dl",
             "kind": "select", "wide": True,
             "options": [(0, "No"), (1, "Yes")]},
            {"name": "RestingECG", "label": "Resting ECG result", "kind": "select",
             "wide": True,
             "options": [(1, "Normal"),
                         (2, "ST-T wave abnormality"),
                         (0, "Left ventricular hypertrophy")]},
        ],
    },
    {
        "title": "During exercise",
        "fields": [
            {"name": "MaxHR", "label": "Maximum heart rate", "kind": "number",
             "unit": "beats/min", "min": 60, "max": 220, "step": 1, "placeholder": "e.g. 150"},
            {"name": "Oldpeak", "label": "ST depression (Oldpeak)", "kind": "number",
             "unit": "mm", "min": -3, "max": 7, "step": 0.1, "placeholder": "e.g. 1.0"},
            {"name": "ExerciseAngina", "label": "Chest pain during exercise",
             "kind": "select", "options": [(0, "No"), (1, "Yes")]},
            {"name": "ST_Slope", "label": "Slope of the peak exercise ST segment",
             "kind": "select",
             "options": [(2, "Upsloping"), (1, "Flat"), (0, "Downsloping")]},
        ],
    },
]

FIELDS = [f for g in GROUPS for f in g["fields"]]

# Column order the model was trained with (independent of the form layout).
FEATURES = (
    list(model.feature_names_in_)
    if hasattr(model, "feature_names_in_")
    else [f["name"] for f in FIELDS]
)

# Fail early on deploy if the form and the pickled model disagree.
assert set(FEATURES) == {f["name"] for f in FIELDS}, (
    "Form fields do not match the model's features: " f"{FEATURES}"
)


def parse_form(form):
    """Validate the submitted form. Returns (values, errors)."""
    values, errors = {}, {}
    for field in FIELDS:
        name = field["name"]
        raw = (form.get(name) or "").strip()
        if raw == "":
            errors[name] = "Required."
            continue
        try:
            number = float(raw)
        except ValueError:
            errors[name] = "Enter a number."
            continue

        if field["kind"] == "number":
            if not field["min"] <= number <= field["max"]:
                errors[name] = f"Enter a value from {field['min']} to {field['max']}."
                continue
            values[name] = number if field["step"] < 1 else int(number)
        else:
            allowed = {code for code, _ in field["options"]}
            if int(number) not in allowed:
                errors[name] = "Choose one of the options."
                continue
            values[name] = int(number)
    return values, errors


def predict(values):
    """Run the KNN model and describe what its nearest neighbours said."""
    row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
    classes = list(model.classes_)
    proba = model.predict_proba(row)[0]
    positive_share = float(proba[classes.index(1)])
    k = int(model.n_neighbors)
    positive_votes = round(positive_share * k)
    prediction = int(model.predict(row)[0])
    return {
        "high_risk": prediction == 1,
        "votes": positive_votes,
        "k": k,
        "percent": round(positive_share * 100),
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart disease risk screen</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      /* ECG paper palette */
      --paper: #FFF9F7;
      --grid-minor: #F8E1DD;
      --grid-major: #F0C4BE;
      --card: #FFFFFF;
      --ink: #1D2B45;
      --ink-soft: #4F5C74;
      --line: #E5D3D0;
      --signal: #C8283C;     /* higher-risk trace */
      --signal-bg: #FCE9EB;
      --calm: #17725F;       /* lower-risk trace */
      --calm-bg: #E4F4EF;

      --font-display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
      --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; }

    html { -webkit-text-size-adjust: 100%; }

    body {
      margin: 0;
      color: var(--ink);
      font: 400 1rem/1.55 var(--font-body);
      background-color: var(--paper);
      background-image:
        linear-gradient(var(--grid-major) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-major) 1px, transparent 1px),
        linear-gradient(var(--grid-minor) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-minor) 1px, transparent 1px);
      background-size: 60px 60px, 60px 60px, 12px 12px, 12px 12px;
      background-attachment: fixed;
    }

    .shell {
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 20px 64px;
      display: grid;
      grid-template-columns: minmax(0, 400px) minmax(0, 1fr);
      gap: 28px 32px;
      align-items: start;
    }

    .intro { grid-column: 1 / -1; max-width: 62ch; }
    .intro h1 {
      margin: 0 0 8px;
      font: 700 clamp(1.9rem, 4vw, 2.7rem)/1.1 var(--font-display);
      letter-spacing: -0.02em;
    }
    .intro p { margin: 0; color: var(--ink-soft); font-size: 1.05rem; }

    /* ---------- Monitor (result panel) ---------- */
    .monitor {
      position: sticky;
      top: 24px;
      background: var(--card);
      border: 2px solid var(--ink);
      border-radius: 14px;
      overflow: hidden;
    }
    .monitor-trace {
      border-bottom: 2px solid var(--ink);
      background-color: var(--card);
      background-image:
        linear-gradient(var(--grid-minor) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-minor) 1px, transparent 1px);
      background-size: 12px 12px;
    }
    .monitor-trace svg { display: block; width: 100%; height: auto; }
    .trace {
      fill: none;
      stroke: var(--ink-soft);
      stroke-width: 2.5;
      stroke-linejoin: round;
      stroke-linecap: round;
    }
    .trace.idle { stroke-dasharray: 2 7; opacity: .7; }
    .monitor.high .trace { stroke: var(--signal); }
    .monitor.low .trace { stroke: var(--calm); }
    .trace.live { stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 1.8s ease-out forwards; }
    @keyframes draw { to { stroke-dashoffset: 0; } }

    .monitor-body { padding: 22px 22px 24px; }
    .monitor-body h2 {
      margin: 0 0 6px;
      font: 700 1.55rem/1.2 var(--font-display);
    }
    .monitor.high h2 { color: var(--signal); }
    .monitor.low h2 { color: var(--calm); }
    .monitor-body p { margin: 0 0 14px; color: var(--ink-soft); }

    .neighbours { display: flex; gap: 8px; margin: 14px 0 8px; }
    .dot {
      width: 26px; height: 26px; border-radius: 50%;
      border: 2px solid var(--ink-soft);
      background: transparent;
    }
    .monitor.high .dot.on { background: var(--signal); border-color: var(--signal); }
    .monitor.low .dot.on { background: var(--signal); border-color: var(--signal); }
    .neighbours-caption { font-size: .92rem; color: var(--ink); margin: 0 0 14px; }

    .note {
      margin: 0;
      padding-top: 14px;
      border-top: 1px solid var(--line);
      font-size: .85rem;
      color: var(--ink-soft);
    }

    /* ---------- Form ---------- */
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px 28px 28px;
    }

    .alert {
      margin: 20px 0 0;
      padding: 12px 14px;
      background: var(--signal-bg);
      border-left: 4px solid var(--signal);
      border-radius: 6px;
      font-weight: 500;
    }

    fieldset { border: 0; margin: 0; padding: 24px 0 4px; min-width: 0; }
    fieldset + fieldset { border-top: 1px solid var(--line); margin-top: 20px; }
    legend {
      padding: 0;
      margin-bottom: 16px;
      font: 700 1.2rem/1.2 var(--font-display);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px 20px;
    }
    .field.wide { grid-column: 1 / -1; }

    label { display: block; font-weight: 600; font-size: .95rem; margin-bottom: 6px; }
    .unit { font-weight: 400; color: var(--ink-soft); }
    .hint { margin: 5px 0 0; font-size: .85rem; color: var(--ink-soft); }

    input, select {
      width: 100%;
      min-height: 46px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
      border: 1.5px solid #B9AAA7;
      border-radius: 8px;
      appearance: none;
      -webkit-appearance: none;
    }
    select {
      padding-right: 38px;
      background-image:
        linear-gradient(45deg, transparent 50%, var(--ink) 50%),
        linear-gradient(135deg, var(--ink) 50%, transparent 50%);
      background-position: calc(100% - 19px) 20px, calc(100% - 13px) 20px;
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
    }
    input::placeholder { color: #8C95A6; }
    input:hover, select:hover { border-color: var(--ink); }
    input:focus-visible, select:focus-visible, button:focus-visible {
      outline: 3px solid var(--ink);
      outline-offset: 2px;
    }
    .field.invalid input, .field.invalid select { border-color: var(--signal); background-color: #FFF7F8; }
    .error { margin: 5px 0 0; font-size: .85rem; font-weight: 500; color: var(--signal); }

    .actions { display: flex; align-items: center; gap: 16px; margin-top: 28px; flex-wrap: wrap; }
    button, .reset {
      font: 600 1rem var(--font-body);
      min-height: 48px;
      padding: 0 26px;
      border-radius: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      text-decoration: none;
    }
    button {
      color: #fff;
      background: var(--ink);
      border: 2px solid var(--ink);
    }
    button:hover { background: #0F1A2E; }
    .reset { color: var(--ink); background: transparent; border: 2px solid var(--line); }
    .reset:hover { border-color: var(--ink); }

    .foot { grid-column: 1 / -1; font-size: .85rem; color: var(--ink-soft); max-width: 70ch; }

    @media (max-width: 880px) {
      .shell { grid-template-columns: minmax(0, 1fr); padding-top: 28px; }
      .monitor { position: static; }
    }
    @media (max-width: 520px) {
      .panel { padding: 4px 18px 22px; }
      .grid { grid-template-columns: minmax(0, 1fr); }
      button, .reset { width: 100%; justify-content: center; }
    }
    @media (prefers-reduced-motion: reduce) {
      .trace.live { animation: none; stroke-dashoffset: 0; }
    }
  </style>
</head>
<body>
  <main class="shell">

    <header class="intro">
      <h1>Heart disease risk screen</h1>
      <p>Enter 11 readings from a recent check-up. The model compares them with {{ n_train }} past patients and reports what the most similar ones had.</p>
    </header>

    <!-- Result monitor -->
    <section class="monitor {% if result %}{{ 'high' if result.high_risk else 'low' }}{% endif %}" id="monitor" aria-live="polite">
      <div class="monitor-trace">
        <svg viewBox="0 0 600 120" role="img" aria-label="{% if result %}Heartbeat trace{% else %}Flat line while waiting for readings{% endif %}">
          {% if result %}
            {% set beat = [(0,60),(50,60),(58,55),(66,60),(84,60),(91,68),(98,14),(105,100),(112,60),(130,60),(146,44),(162,60),(200,60)] %}
            <path class="trace live" pathLength="1" d="M0,60{% for i in range(3) %}{% for p in beat %} L{{ p[0] + i * 200 }},{{ p[1] }}{% endfor %}{% endfor %}"/>
          {% else %}
            <path class="trace idle" d="M0,60 L600,60"/>
          {% endif %}
        </svg>
      </div>

      <div class="monitor-body">
        {% if result %}
          <h2>{{ 'Higher risk pattern' if result.high_risk else 'Lower risk pattern' }}</h2>
          <p>
            {% if result.high_risk %}
              Most of the patients with readings closest to yours had heart disease.
            {% else %}
              Most of the patients with readings closest to yours did not have heart disease.
            {% endif %}
          </p>
          <div class="neighbours" role="img" aria-label="{{ result.votes }} of {{ result.k }} similar patients had heart disease">
            {% for i in range(result.k) %}<span class="dot {{ 'on' if i < result.votes else '' }}"></span>{% endfor %}
          </div>
          <p class="neighbours-caption">{{ result.votes }} of the {{ result.k }} most similar patients had heart disease.</p>
        {% else %}
          <h2>Waiting for readings</h2>
          <p>Fill in the form and select <strong>Check risk</strong>. Your result appears here.</p>
        {% endif %}
        <p class="note">This is a screening estimate from a statistical model, not a diagnosis. Share your readings with a doctor for medical advice.</p>
      </div>
    </section>

    <!-- Form -->
    <form class="panel" method="post" action="{{ url_for('index') }}" novalidate>
      {% if errors %}
        <div class="alert" role="alert">
          {{ errors|length }} {{ 'field needs' if errors|length == 1 else 'fields need' }} attention before the check can run.
        </div>
      {% endif %}

      {% for group in groups %}
      <fieldset>
        <legend>{{ group.title }}</legend>
        <div class="grid">
          {% for f in group.fields %}
            {% set err = errors.get(f.name) %}
            <div class="field {{ 'wide' if f.wide else '' }} {{ 'invalid' if err else '' }}">
              <label for="{{ f.name }}">{{ f.label }}{% if f.unit %} <span class="unit">({{ f.unit }})</span>{% endif %}</label>

              {% if f.kind == 'number' %}
                <input id="{{ f.name }}" name="{{ f.name }}" type="number" inputmode="decimal"
                       min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                       placeholder="{{ f.placeholder }}" value="{{ form.get(f.name, '') }}"
                       required {% if err %}aria-invalid="true" aria-describedby="{{ f.name }}-msg"{% endif %}>
              {% else %}
                <select id="{{ f.name }}" name="{{ f.name }}" required
                        {% if err %}aria-invalid="true" aria-describedby="{{ f.name }}-msg"{% endif %}>
                  <option value="" disabled {{ 'selected' if not form.get(f.name) else '' }}>Select an option</option>
                  {% for code, text in f.options %}
                    <option value="{{ code }}" {{ 'selected' if form.get(f.name) == code|string else '' }}>{{ text }}</option>
                  {% endfor %}
                </select>
              {% endif %}

              {% if err %}
                <p class="error" id="{{ f.name }}-msg">{{ err }}</p>
              {% elif f.hint %}
                <p class="hint">{{ f.hint }}</p>
              {% endif %}
            </div>
          {% endfor %}
        </div>
      </fieldset>
      {% endfor %}

      <div class="actions">
        <button type="submit">Check risk</button>
        <a class="reset" href="{{ url_for('index') }}">Clear form</a>
      </div>
    </form>

    <p class="foot">The model is a 5-nearest-neighbours classifier. It was trained on a limited dataset and has no knowledge of your wider health history.</p>
  </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    form, errors, result = {}, {}, None

    if request.method == "POST":
        form = request.form.to_dict()
        values, errors = parse_form(request.form)
        if not errors:
            result = predict(values)

    return render_template_string(
        PAGE,
        groups=GROUPS,
        form=form,
        errors=errors,
        result=result,
        n_train=int(getattr(model, "n_samples_fit_", 0)),
    )


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
