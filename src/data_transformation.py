import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def transform_data(file_path):

    """
    Transforms the data for machine learning.
        Args:
        file_path (str): The path to the CSV file containing the data.
    Returns:
        tuple: A tuple containing the transformed training and test sets. as (X_train, X_test, y_train, y_test)
    """

    # load data
    df = pd.read_csv(file_path)

    # separate target
    X = df.drop("Personal Loan", axis=1)
    y = df["Personal Loan"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test