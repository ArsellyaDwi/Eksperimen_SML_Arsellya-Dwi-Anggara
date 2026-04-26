import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =========================
# CONNECT DAGSHUB
# =========================
dagshub.init(repo_owner='ArsellyaDwi', repo_name='titanic-mlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/ArsellyaDwi/titanic-mlflow.mlflow")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("titanic_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================
# TRAINING + TUNING
# =========================
n_estimators_list = [50, 100]
max_depth_list = [3, 5]

for n in n_estimators_list:
    for d in max_depth_list:

        with mlflow.start_run():

            model = RandomForestClassifier(n_estimators=n, max_depth=d)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            # =========================
            # LOG PARAM
            # =========================
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)

            # =========================
            # LOG METRICS
            # =========================
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            # =========================
            # LOG MODEL
            # =========================
            mlflow.sklearn.log_model(model, "model")

            # =========================
            # ARTIFACT TAMBAHAN (WAJIB ADVANCED)
            # =========================

            # 1. Save feature importance plot
            plt.figure()
            plt.barh(X.columns, model.feature_importances_)
            plt.title("Feature Importance")
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")

            # 2. Save metrics ke file
            with open("metrics.txt", "w") as f:
                f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}")

            mlflow.log_artifact("metrics.txt")