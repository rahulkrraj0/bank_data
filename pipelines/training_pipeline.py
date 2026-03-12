# pipelines/training_pipeline.py
import sys
import os
import mlflow
import skops.io as sio


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_transformation import transform_data
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model
from src.utils.common import save_object

def run_training_pipeline():

    """python pipelines/training_pipeline.py"""

    file_path = r"data\processed\bpl_data_cleaned.csv"

    X_train, X_test, y_train, y_test = transform_data(file_path)

    trained_models = train_model(X_train, y_train)

    best_model = None
    best_score = 0

    # Create a directory to store skops models
    os.makedirs("skops_models", exist_ok=True)

    for name, model in trained_models.items():

        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name=name):

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print(f"\nModel: {name}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
            # Save model with skops and log as artifact
            skops_path = os.path.join("skops_models", f"{name}.skops")
            sio.dump(model, skops_path)
            mlflow.log_artifact(skops_path, artifact_path="skops_model")
            

        # select best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    # log best model
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_metric("accuracy", best_score)
        
        # Save best model with skops and log as artifact
        skops_path = os.path.join("skops_models", "best_model.skops")
        sio.dump(best_model, skops_path)
        mlflow.log_artifact(skops_path, artifact_path="best_model_skops")
    

# if __name__ == "__main__":
run_training_pipeline()
