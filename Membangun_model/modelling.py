import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# load data
df = pd.read_csv("titanic_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow local
mlflow.set_tracking_uri("file:./mlruns")

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)