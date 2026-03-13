# pipelines/training_pipeline.py
import sys
import os
import mlflow
import mlflow.sklearn
import pickle


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

    for name, model in trained_models.items():

        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name=name):

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print(f"\nModel: {name}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
            mlflow.sklearn.log_model(model, name)
            

        # select best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    # log best model
    mlflow.sklearn.log_model(best_model, "best_model")

    # Save the best model as a pickle file
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    

# if __name__ == "__main__":
run_training_pipeline()
