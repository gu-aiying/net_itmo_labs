import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

def test_registered_models():
    client = MlflowClient()

    print("ТЕСТИРОВАНИЕ ЗАРЕГИСТРИРОВАННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    models = client.search_registered_models()

    for model in models:
        print(f"\nМодель: {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")

        for version in versions:
            print(f"\n Версия {version.version}:")
            try:
                model_uri = f"models:/{model.name}/{version.version}"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                print(" Модель загружена успешно")

                test_data = [[5.1, 3.5, 1.4, 0.2]]

                prediction = loaded_model.predict(test_data)
                print(f" Предсказание работает: класс {prediction[0]}")

            except Exception as e:
                print(f" Ошибка: {e}")

if __name__ == "__main__":
    test_registered_models()