import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели из локальной папки
try:
    logger.info("Loading model from local directory...")
    
    # Пробуем загрузить через joblib (лучше для sklearn)
    try:
        import joblib
        model = joblib.load('/app/model/model.joblib')
        logger.info("Model loaded successfully using joblib")
    except:
        # Если joblib не работает, используем pickle
        with open('/app/model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully using pickle")
    
    # Загружаем информацию о модели
    with open('/app/model/model_info.json', 'r') as f:
        model_info = json.load(f)
        
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

app = FastAPI(
    title="ML Inference Service", 
    version="1.0.0",
    description="Service for ML model predictions"
)

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list
    model_version: str
    model_accuracy: float

@app.get("/")
async def root():
    return {
        "message": "ML Inference Service for Classification", 
        "status": "healthy",
        "model": "RandomForest Classifier",
        "features": model_info["n_features"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Валидация количества фичей
        expected_features = model_info["n_features"]
        if len(request.features) != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_features} features, got {len(request.features)}"
            )
        
        # Преобразование в DataFrame для модели
        input_data = pd.DataFrame([request.features])
        
        # Предсказание
        prediction = model.predict(input_data)
        
        # Получение вероятностей
        try:
            probabilities = model.predict_proba(input_data)
            probabilities_list = probabilities[0].tolist()
        except Exception:
            probabilities_list = [1.0 if pred == prediction[0] else 0.0 for pred in [0, 1]]
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probabilities=probabilities_list,
            model_version="RandomForest-v1",
            model_accuracy=model_info["test_accuracy"]
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info_endpoint():
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)