# SalaryPredictions

This repository now has a clean split between model creation and model hosting.

## Folder Structure

- `ML_Model/`: notebook, training dependencies, and export logic for creating the final CatBoost model.
- `AppHosting/`: the containerized inference API meant to run on your QNAP through Container Station.

## Deployment Flow

1. Train and export the final model from `ML_Model/`.
2. Copy the exported files into `AppHosting/model_artifacts/`.
3. Build and run the container from `AppHosting/`.
4. Expose port `8000` on your LAN and access it internally.

## Internal API

- `GET /health`
- `POST /predict`

Example request:

```json
{
  "job_title": "Data Scientist",
  "education_level": "Master",
  "company_size": "Large",
  "location": "Canada",
  "remote_work": "Hybrid",
  "experience_years": 6,
  "skills_count": 12,
  "certifications": 2
}
```
