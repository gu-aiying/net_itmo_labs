import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import sklearn
import os

mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

configs = [
{"n_estimators": 50, "max_depth": 3, "run_name": "Small RF"},
{"n_estimators": 100, "max_depth": 5, "run_name": "Medium RF"},
{"n_estimators": 200, "max_depth": 10, "run_name": "Large RF"},
{"n_estimators": 100, "max_depth": None, "run_name": "Default RF"}
]

for config in configs:
    with mlflow.start_run(run_name=config["run_name"]):
        # Логируем версии библиотек
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)

        # Обучение модели
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
        )

        mlflow.log_params(config)

        mlflow.log_param("random_state", 42)

        model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
        })
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(
        model_uri=model_uri,
        name="Iris_RandomForest"
        )

        print(f"Запуск '{config['run_name']}' завершен. Accuracy: {accuracy:.4f}")

print("Все запуски завершены и модели зарегистрированы!")