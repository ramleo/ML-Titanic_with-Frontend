#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import joblib, pandas as pd, io, json, os

app = FastAPI(title="ML Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = joblib.load("models/final_pipeline.pkl")
_fi_file = "models/feature_importance.json"
_feature_importance = json.load(open(_fi_file)) if os.path.exists(_fi_file) else []
_ranges   = json.load(open("models/feature_ranges.json")) if os.path.exists("models/feature_ranges.json") else {}
label_encoder = joblib.load("models/label_encoder.pkl")

class InputData(BaseModel):
    Pclass: Optional[float] = None
    Age: Optional[float] = None
    SibSp: Optional[float] = None
    Parch: Optional[float] = None
    Fare: Optional[float] = None
    Name: Optional[str] = None
    Sex: Optional[str] = None
    Ticket: Optional[str] = None
    Embarked: Optional[str] = None

_ID_COLS = ["PassengerId"]  # pipeline trained with this; injected as NaN at predict time

@app.get("/")
def index():
    """Serve the prediction UI if index.html exists."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "ML Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    for col in _ID_COLS:
        df[col] = float('nan')
    pred = pipeline.predict(df)[0]
    label = label_encoder.inverse_transform([pred])[0]
    proba = pipeline.predict_proba(df)[0].tolist()
    return {"prediction": str(label), "probabilities": proba,
            "feature_importance": _feature_importance}

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.DataFrame([d.dict() for d in data])
    for col in _ID_COLS:
        df[col] = float('nan')
    preds = pipeline.predict(df)
    labels = label_encoder.inverse_transform(preds).tolist()
    return {"predictions": [str(l) for l in labels]}

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Upload a CSV — returns predictions for every row."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")
    contents = await file.read()
    df_up = pd.read_csv(io.BytesIO(contents))
    preds_up = pipeline.predict(df_up)
    try:
        labels_up = label_encoder.inverse_transform(preds_up).tolist()
    except Exception:
        labels_up = [str(p) for p in preds_up]
    return {"count": len(labels_up), "predictions": labels_up}


@app.get("/ranges")
def ranges_endpoint():
    return _ranges

@app.get("/importance")
def importance():
    return {"feature_importance": _feature_importance}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
