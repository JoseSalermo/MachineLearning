# AppHosting

This folder is the hosting side of the project.

Use it when you want to run the final trained salary model as an internal API on your QNAP NAS through Container Station.

## Expected Inputs

Place these exported files in `model_artifacts/`:

- `salary_model.cbm`
- `metadata.json`

Those files should be created from the training/export workflow under `ML_Model/`.

## Container Service

The API exposes:

- `GET /health`
- `POST /predict`

The container listens on port `8000`.
