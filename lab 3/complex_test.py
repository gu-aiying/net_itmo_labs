import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

def comprehensive_model_test(loaded_model, model_name, version):
    """Комплексное тестирование модели"""
    iris = load_iris()

    print(f"\nКОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ: {model_name} v{version}")
    print("=" * 60)

    # ТЕСТ 1: Базовый тест на четких примерах
    print("\nТЕСТ 1: Базовый тест (четкие примеры)")
    clear_examples = [
        ([5.1, 3.5, 1.4, 0.2], 0, "setosa"),
        ([6.0, 2.7, 5.1, 1.6], 1, "versicolor"),
        ([6.7, 3.0, 5.2, 2.3], 2, "virginica")
    ]

    correct_predictions = 0
    for features, true_class, expected_name in clear_examples:
        prediction = loaded_model.predict([features])[0]
    is_correct = (prediction == true_class)
    if is_correct:
        correct_predictions += 1

    status = "√" if is_correct else "×"
    print(f" {status} {features}")
    print(f" Ожидалось: {expected_name}")
    print(f" Предсказано: {iris.target_names[prediction]}")

    accuracy_clear = correct_predictions / len(clear_examples)
    print(f" Точность на четких примерах: {accuracy_clear:.4f}")

    # ТЕСТ 2: Тест на пограничных случаях
    print("\n ТЕСТ 2: Пограничные случаи")
    borderline_cases = [

        ([5.8, 2.7, 3.9, 1.2], "setosa/versicolor"),
        ([6.2, 2.9, 4.3, 1.3], "versicolor/virginica"),
        ([5.5, 2.4, 3.8, 1.1], "setosa/versicolor"),
        ([6.4, 3.2, 5.3, 2.3], "versicolor/virginica")
    ]

    print(" Анализ уверенности на пограничных случаях:")
    for features, description in borderline_cases:
        try:
            probabilities = loaded_model.predict_proba([features])[0]
            prediction = loaded_model.predict([features])[0]

            max_prob = max(probabilities)
            confidence_level = "высокая" if max_prob > 0.8 else "средняя" if max_prob > 0.6 else "низкая"

            print(f" {description}: {features}")
            print(f" Предсказание: {iris.target_names[prediction]}")
            print(f" Уверенность: {confidence_level} ({max_prob:.3f})")
            print(f" Распределение: {[f'{p:.3f}' for p in probabilities]}")
        except Exception as e:
            print(f" Ошибка: {e}")

    # ТЕСТ 3: Тест на большом наборе данных
    print("\n ТЕСТ 3: Производительность на тестовом наборе")
    # Используем встроенный набор данных для тестирования

    X_test = iris.data[120:140]  # Последние 20 samples
    y_test = iris.target[120:140]

    predictions = loaded_model.predict(X_test)
    accuracy_large = accuracy_score(y_test, predictions)

    # Подсчет правильных предсказаний по классам
    correct_by_class = {0: 0, 1: 0, 2: 0}
    total_by_class = {0: 0, 1: 0, 2: 0}

    for true, pred in zip(y_test, predictions):
        total_by_class[true] += 1
    if true == pred:
        correct_by_class[true] += 1

    print(f" Общая точность: {accuracy_large:.4f}")
    print(" Точность по классам:")
    for class_id in range(3):
        if total_by_class[class_id] > 0:
            class_accuracy = correct_by_class[class_id] / total_by_class[class_id]
            print(f" {iris.target_names[class_id]}: {class_accuracy:.4f} "
                  f"({correct_by_class[class_id]} / {total_by_class[class_id]})")

    # ТЕСТ 4: Анализ метрик
    print("\n📋 ТЕСТ 4: Детальные метрики")
    try:
        report = classification_report(y_test, predictions, target_names=iris.target_names,

                                       output_dict=False)

        print(" Отчет классификации:")
        for line in report.split('\n'):
            if line.strip():
                print(f" {line}")
    except:
        print(" Не удалось сгенерировать детальный отчет")

    # ТЕСТ 5: Проверка консистентности
    print("\n ТЕСТ 5: Проверка консистентности")
    test_sample = [5.1, 3.5, 1.4, 0.2]
    predictions_same = []

    # Несколько предсказаний на одном примере
    for i in range(3):
        pred = loaded_model.predict([test_sample])[0]
    predictions_same.append(pred)

    is_consistent = all(p == predictions_same[0] for p in predictions_same)
    consistency_status = "√ Консистентна" if is_consistent else "× Не консистентна"
    print(f" Модель {consistency_status}")
    print(f" Предсказания на одном примере: {[iris.target_names[p] for p in

                                              predictions_same]}")

    return {
        'accuracy_clear': accuracy_clear,
        'accuracy_large': accuracy_large,
        'is_consistent': is_consistent,

        'correct_clear': f"{correct_predictions}/{len(clear_examples)}"
    }


def run_comprehensive_testing():
    """Запуск комплексного тестирования всех моделей"""
    client = MlflowClient()

    print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ЗАРЕГИСТРИРОВАННЫХ МОДЕЛЕЙ")
    print("=" * 70)

    models = client.search_registered_models()
    all_results = []

    for model in models:
        print(f"\n МОДЕЛЬ: {model.name}")
        print("-" * 50)

        versions = client.search_model_versions(f"name='{model.name}'")

        for version in versions:
            print(f"\n Версия {version.version}")
            print(f" Run ID: {version.run_id}")

            try:
                # Загружаем модель
                model_uri = f"models:/{model.name}/{version.version}"
                loaded_model = mlflow.sklearn.load_model(model_uri)

                print(" Модель загружена")

                # Запускаем комплексное тестирование
                test_results = comprehensive_model_test(loaded_model, model.name,

                version.version)

                # Сохраняем результаты
                all_results.append({
                'model': model.name,
                'version': version.version,
                'run_id': version.run_id,
                'results': test_results
                })

            except Exception as e:
                print(f" Ошибка: {e}")

    # Сравнительный анализ
    if all_results:
        print(f"\n{'='*80}")
        print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
        print(f"{'='*80}")

        # Сортировка по точности на большом наборе
        all_results.sort(key=lambda x: x['results']['accuracy_large'], reverse=True)

        print("\nРейтинг моделей по результатам тестирования:")

        for i, result in enumerate(all_results, 1):
            print(f"{i}. {result['model']} v{result['version']}")

            print(f" Четкие примеры: {result['results']['accuracy_clear']:.4f} ({result['results']['correct_clear']})")

            print(f" Большой набор: {result['results']['accuracy_large']:.4f}")
            print(f" Консистентность: {' Да' if result['results']['is_consistent'] else ' Нет'}")
            print(f" Run ID: {result['run_id'][:12]}...")
            print()

if __name__ == "__main__":
    run_comprehensive_testing()

    print("\n ТЕСТИРОВАНИЕ ЗАВЕРШЕНО:")
    print(" - Проведено 5 различных тестов на каждой модели")
    print(" - Проанализирована точность на разных типах данных")
    print(" - Проверена уверенность модели на пограничных случаях")
    print(" - Оценена консистентность предсказаний")
    print(" - Сравнены все версии моделей")