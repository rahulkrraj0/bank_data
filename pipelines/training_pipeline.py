# pipelines/training_pipeline.py
import sys
import os
import mlflow
import mlflow.sklearn


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_transformation import transform_data
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model
from src.utils.common import save_object

def run_training_pipeline():

    """python pipelines/training_pipeline.py"""

    file_path = r"data\processed\bpl_data_cleaned.csv"

    X_train, X_test, y_train, y_test = transform_data(file_path)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    mlflow.sklearn.log_model(model, "new model")

# if __name__ == "__main__":
run_training_pipeline()