# Bank Loan Prediction

This project is an end-to-end machine learning project that predicts whether a person will get a bank loan or not. It uses MLFlow for experiment tracking and DVC for data version control.

## Overview

This project is designed to predict bank loan eligibility. It includes data cleaning, model training, and a prediction pipeline. The project is structured to be scalable and maintainable, using MLOps best practices.

## Technologies Used

*   Python
*   Pandas
*   Scikit-learn
*   MLflow
*   DVC
*   FastAPI
*   Jinja2

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MLFlow
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training Pipeline

To run the training pipeline, execute the following command:

```bash
python pipelines/training_pipeline.py
```

This will train the models specified in `config/config.yaml` and log the experiments to MLFlow.

### Prediction Pipeline

To make predictions, use the prediction pipeline:

```bash
python pipelines/prediction_pipeline.py --data <path-to-your-data>
```

## Project Structure

```
├── .dvc/
├── .dvcignore
├── .git/
├── .gitignore
├── .venv/
├── config/
│   └── config.yaml
├── data/
├── data.dvc
├── dvc_storage/
├── mlflow.ipynb
├── mlruns/
├── models/
│   └── model.pkl
├── notebooks/
│   └── data_cleaning.ipynb
├── pipelines/
│   ├── prediction_pipeline.py
│   └── training_pipeline.py
├── prediction.py
├── README.md
├── requirements.txt
└── src/
    ├── data_transformation.py
    ├── model_evaluation.py
    ├── model_trainer.py
    └── utils/
        ├── common.py
        ├── delete_model.py
        ├── exception.py
        └── logger.py
```

## MLflow Tracking

This project uses MLflow for experiment tracking. The MLflow UI can be launched with the following command:

```bash
mlflow ui
```

The UI will be available at `http://127.0.0.1:5000` by default. The experiments are logged under the experiment name "bank_loan".

## DVC

Data Version Control (DVC) is used to manage the data. To pull the data, run:

```bash
dvc pull
```

## WebApp using FastAPI

This projech have app.py file which is use to run the web app. To run the appuvicorn app.py:

```bash
    uvicorn app:app --reload

```


