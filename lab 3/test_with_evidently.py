import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")

def load_production_model():
    """Загрузка продакшен модели из MLflow Registry"""
    print(" Загрузка модели из MLflow Registry...")

    try:
        client = mlflow.tracking.MlflowClient()
        print(" MLflow client создан")

        model_name = "Iris_RandomForest"
        print(f" Поиск модели: {model_name}")

        versions = client.search_model_versions(f"name='{model_name}'")
        print(f" Найдено версий: {len(versions)}")

        if not versions:
            raise Exception(f"Модель {model_name} не найдена в registry")

        for v in versions:
            print(f" Версия {v.version}: {v.status} -> {v.current_stage}")

        production_versions = [v for v in versions if v.current_stage == "Production"]

        if production_versions:
            target_version = sorted(production_versions, key=lambda x: int(x.version))[-1]
            print(f" Используем продакшен версию: {target_version.version}")
        else:
            target_version = sorted(versions, key=lambda x: int(x.version))[-1]
            print(f" Продакшен не найден, используем последнюю версию: {target_version.version}")

        model_uri = f"models:/{model_name}/{target_version.version}"
        print(f"URI модели: {model_uri}")

        print(" Загрузка модели...")
        model = mlflow.sklearn.load_model(model_uri)
        print(" Модель успешно загружена!")

        return model

    except Exception as e:
        print(f" Ошибка при загрузке модели: {e}")
        raise

def prepare_test_data():
    """Подготовка тестовых данных"""

    print("\n Подготовка тестовых данных...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Разделяем на референсные и текущие данные
    X_ref, X_curr, y_ref, y_curr = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    reference_data = pd.DataFrame(X_ref, columns=feature_names)
    reference_data['target'] = y_ref

    current_data = pd.DataFrame(X_curr, columns=feature_names)
    current_data['target'] = y_curr

    print(f" Референсные данные: {reference_data.shape}")
    print(f" Текущие данные: {current_data.shape}")

    return reference_data, current_data, feature_names

def run_evidently_analysis(model, reference_data, current_data, feature_names):
    """Запуск анализа с Evidently"""
    print("\n Запуск анализа с Evidently...")

    # Добавляем предсказания (используем numpy arrays чтобы избежать warning)
    reference_data['prediction'] = model.predict(reference_data[feature_names].values)

    current_data['prediction'] = model.predict(current_data[feature_names].values)

    # 1. Анализ дрейфа данных
    print(" Анализ дрейфа данных...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_data[feature_names],
        current_data=current_data[feature_names]
    )
    drift_report.save_html("data_drift_report.html")
    print(" Отчет по дрейфу сохранен: data_drift_report.html")

    # Безопасное извлечение результатов дрейфа
    drift_results = drift_report.as_dict()
    drift_detected = False
    drifted_features = 0

    for metric in drift_results['metrics']:
        if 'dataset_drift' in metric['result']:
            drift_detected = metric['result']['dataset_drift']
        if 'number_of_drifted_features' in metric['result']:
            drifted_features = metric['result']['number_of_drifted_features']
        elif 'n_drifted_features' in metric['result']:
            drifted_features = metric['result']['n_drifted_features']

    print(f" Дрейф данных: {' ОБНАРУЖЕН' if drift_detected else ' НЕТ'}")
    print(f" Признаков с дрейфом: {drifted_features}")

    # 2. Анализ качества модели
    print(" Анализ качества модели...")
    model_report = Report(metrics=[ClassificationPreset()])
    model_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    model_report.save_html("model_quality_report.html")
    print(" Отчет по качеству модели сохранен: model_quality_report.html")

    # Безопасное извлечение метрик качества
    model_results = model_report.as_dict()
    accuracy_curr = 0

    for metric in model_results['metrics']:
        if 'current' in metric['result'] and 'accuracy' in metric['result']['current']:
            accuracy_curr = metric['result']['current']['accuracy']
            break
        elif 'accuracy' in metric['result']:
            accuracy_curr = metric['result']['accuracy']
            break

    print(f" Accuracy: {accuracy_curr:.4f}")

    return drift_detected, drifted_features, accuracy_curr

def main():
    """Основная функция"""
    print(" ТЕСТИРОВАНИЕ МОДЕЛИ С EVIDENTLY")
    print("=" * 50)

    try:
        # Загружаем модель из MLflow
        model = load_production_model()

        # Подготавливаем данные
        reference_data, current_data, feature_names = prepare_test_data()

        # Запускаем анализ с Evidently
        drift_detected, drifted_features, accuracy = run_evidently_analysis(
            model, reference_data, current_data, feature_names
        )

        # Вывод результатов
        print("\n" + "=" * 60)
        print(" РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        print(f" Data Drift: {' ОБНАРУЖЕН' if drift_detected else ' НЕТ'}")
        print(f" Accuracy: {accuracy:.4f}")

        if drift_detected:
            print(f" Дрейфующих признаков: {drifted_features}")
        print(f" Отчеты:")
        print(f" - data_drift_report.html")

        print(f" - model_quality_report.html")
        print("\n ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

    except Exception as e:
        print(f" Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()