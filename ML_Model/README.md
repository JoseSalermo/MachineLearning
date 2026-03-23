# MachineLearning
My Personal Projects for Learning

## Python setup

This repo uses a local virtual environment in `.venv`.

Create it with `python3` because some systems do not provide a `python` command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name machinelearning --display-name "Python (MachineLearning)"
```

If you open `SalaryPredictions.ipynb`, select the `Python (MachineLearning)` kernel.
