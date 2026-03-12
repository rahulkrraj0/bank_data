import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}




def train_model(X_train, y_train, models=models):


    mlflow.set_experiment("bank_loan_prediction")

    trained_models = {}

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            # log parameters
            mlflow.log_param("model", name)

            # train model
            model.fit(X_train, y_train)

            # log model
            mlflow.sklearn.log_model(
                sk_model=model,
                name=name
            )

            trained_models[name] = model

    return trained_models