import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    mlflow.set_experiment("bank_loan_prediction")

    with mlflow.start_run():

        # log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # train model
        model.fit(X_train, y_train)


        # log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="loan_model"
            
        )

    return model