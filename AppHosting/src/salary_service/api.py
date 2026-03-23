from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model_artifacts"))

FIELD_LABELS = {
    "job_title": "Job Title",
    "education_level": "Education Level",
    "company_size": "Company Size",
    "location": "Location",
    "remote_work": "Remote Work",
    "experience_years": "Experience Years",
    "skills_count": "Skills Count",
    "certifications": "Certifications",
}
DEFAULT_VALUES = {
    "job_title": "Data Scientist",
    "education_level": "Master",
    "company_size": "Large",
    "location": "Canada",
    "remote_work": "Hybrid",
    "experience_years": 6,
    "skills_count": 12,
    "certifications": 2,
}


class SalaryPredictionRequest(BaseModel):
    job_title: str = Field(..., examples=["Data Scientist"])
    education_level: str = Field(..., examples=["Master"])
    company_size: str = Field(..., examples=["Large"])
    location: str = Field(..., examples=["Canada"])
    remote_work: str = Field(..., examples=["Hybrid"])
    experience_years: float = Field(..., ge=0, examples=[6])
    skills_count: int = Field(..., ge=0, examples=[12])
    certifications: int = Field(..., ge=0, examples=[2])


def load_model_artifacts(model_dir: Path) -> tuple[CatBoostRegressor, dict]:
    model = CatBoostRegressor()
    model.load_model(model_dir / "salary_model.cbm")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def build_inference_frame(payload: dict, metadata: dict) -> pd.DataFrame:
    frame = pd.DataFrame([payload], columns=metadata["feature_columns"])
    for col in metadata["categorical_features"]:
        frame[col] = frame[col].astype(str)
    return frame


def render_select(name: str, options: list[str], selected_value: str) -> str:
    option_tags = []
    for option in options:
        selected = " selected" if option == selected_value else ""
        escaped = escape(str(option))
        option_tags.append(f'<option value="{escaped}"{selected}>{escaped}</option>')

    return (
        f"<label>{FIELD_LABELS[name]}"
        f'<select name="{name}">{"".join(option_tags)}</select>'
        "</label>"
    )


def render_text_input(name: str, value: str) -> str:
    return (
        f"<label>{FIELD_LABELS[name]}"
        f'<input name="{name}" value="{escape(value)}" required>'
        "</label>"
    )


def render_numeric_input(name: str, value: int | float, step: str) -> str:
    return (
        f"<label>{FIELD_LABELS[name]}"
        f'<input name="{name}" type="number" min="0" step="{step}" value="{value}" required>'
        "</label>"
    )


def render_index_html(metadata: dict) -> str:
    category_options = metadata.get("category_options", {})
    categorical_fields = []

    for name in ["job_title", "education_level", "company_size", "location", "remote_work"]:
        options = category_options.get(name)
        if options:
            categorical_fields.append(render_select(name, options, str(DEFAULT_VALUES[name])))
        else:
            categorical_fields.append(render_text_input(name, str(DEFAULT_VALUES[name])))

    numeric_fields = [
        render_numeric_input("experience_years", DEFAULT_VALUES["experience_years"], "0.1"),
        render_numeric_input("skills_count", DEFAULT_VALUES["skills_count"], "1"),
        render_numeric_input("certifications", DEFAULT_VALUES["certifications"], "1"),
    ]

    form_fields = "".join(categorical_fields + numeric_fields)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Salary Prediction</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe6;
      --panel: #fffaf2;
      --line: #d7c8b2;
      --text: #2f2419;
      --muted: #6f6254;
      --accent: #0e7490;
      --accent-strong: #155e75;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff7eb 0, transparent 28rem),
        linear-gradient(180deg, #efe4d1 0%, var(--bg) 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3.5rem);
      margin: 0 0 8px;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 24px;
      line-height: 1.5;
    }}
    .card {{
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 14px 40px rgba(47, 36, 25, 0.08);
    }}
    form {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    label {{
      display: grid;
      gap: 8px;
      font-size: 0.95rem;
    }}
    input, select, button {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }}
    button {{
      background: var(--accent);
      color: white;
      border: 0;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .full {{ grid-column: 1 / -1; }}
    .result {{
      margin-top: 20px;
      padding: 18px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid var(--line);
      display: none;
    }}
    .salary {{
      font-size: clamp(1.8rem, 4vw, 3rem);
      margin: 8px 0 0;
    }}
    .error {{
      color: #991b1b;
      margin-top: 12px;
      display: none;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Salary Prediction</h1>
    <p>Fill in a profile and estimate salary from the model hosted on this server.</p>
    <section class="card">
      <form id="prediction-form">
        {form_fields}
        <div class="full">
          <button type="submit">Predict Salary</button>
        </div>
      </form>
      <div id="result" class="result">
        <div>Predicted salary</div>
        <div id="salary" class="salary"></div>
      </div>
      <div id="error" class="error"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("prediction-form");
    const result = document.getElementById("result");
    const salary = document.getElementById("salary");
    const errorBox = document.getElementById("error");

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      result.style.display = "none";
      errorBox.style.display = "none";
      errorBox.textContent = "";

      const formData = new FormData(form);
      const payload = {{
        job_title: formData.get("job_title"),
        education_level: formData.get("education_level"),
        company_size: formData.get("company_size"),
        location: formData.get("location"),
        remote_work: formData.get("remote_work"),
        experience_years: Number(formData.get("experience_years")),
        skills_count: Number(formData.get("skills_count")),
        certifications: Number(formData.get("certifications"))
      }};

      try {{
        const response = await fetch("/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
        const data = await response.json();

        if (!response.ok) {{
          throw new Error(data.detail || "Prediction failed");
        }}

        salary.textContent = `$${{data.predicted_salary.toLocaleString()}}`;
        result.style.display = "block";
      }} catch (error) {{
        errorBox.textContent = error.message;
        errorBox.style.display = "block";
      }}
    }});
  </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, metadata = load_model_artifacts(MODEL_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Expected model_artifacts in {MODEL_DIR}. "
            "Export the final model from ML_Model first, then mount that folder here."
        ) from exc

    app.state.model = model
    app.state.metadata = metadata
    yield


app = FastAPI(title="Salary Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(render_index_html(app.state.metadata))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SalaryPredictionRequest) -> dict:
    payload = request.model_dump()

    try:
        frame = build_inference_frame(payload, app.state.metadata)
        prediction = float(app.state.model.predict(frame)[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "predicted_salary": round(prediction, 2),
        "currency": "USD",
        "input": payload,
    }
