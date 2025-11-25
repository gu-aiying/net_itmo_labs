import requests
import json
import time

# URL сервиса (внутри кластера)
SERVICE_URL = "http://ml-inference-service:8000"


def test_service():
    print("Testing ML Inference Service...")

    # Тест здоровья
    try:
        health = requests.get(f"{SERVICE_URL}/health")
        print(f" Health check: {health.status_code} - {health.json()}")
    except Exception as e:
        print(f" Health check failed: {e}")
        return

    # Тест информации о модели
    try:
        info = requests.get(f"{SERVICE_URL}/model-info")
        print(f" Model info: {info.status_code}")
        print(json.dumps(info.json(), indent=2))
    except Exception as e:
        print(f" Model info failed: {e}")

    # Тест предсказания
    test_data = {
        "features": [0.1, -0.5, 1.2, -0.8, 0.3, 0.7, -1.1, 0.9, -0.2, 0.4,
                     0.6, -0.3, 1.0, -0.7, 0.2, 0.8, -0.9, 0.5, -0.1, 0.3,
                     0.1, -0.4, 0.9, -0.6, 0.2]  # 25 фичей
    }

    try:
        start_time = time.time()
        response = requests.post(f"{SERVICE_URL}/predict", json=test_data)
        end_time = time.time()

        print(f" Prediction: {response.status_code}")
        print(f"  Response time: {(end_time - start_time) * 1000:.2f}ms")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f" Prediction failed: {e}")


if __name__ == "__main__":
    test_service()