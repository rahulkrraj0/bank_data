import pandas as pd
from pipelines.prediction_pipeline import predict

data = pd.DataFrame({
    "Age":[40],
    "Experience":[10],
    "Income":[60],
    "Family":[2],
    "CCAvg":[1.5],
    "Education":[2],
    "Mortgage":[0],
    "Securities Account":[1],
    "CD Account":[0],
    "Online":[1],
    "CreditCard":[0]
})

print(predict(data))