import os
import sys
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.exception import NetworkSecurityException
from src.logger import logging
from src.utils import load_object, NetworkModel
from src.pipeline import TrainingPipeline

app = FastAPI(title="Network Security - Phishing Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

os.makedirs("prediction_output", exist_ok=True)
os.makedirs("models", exist_ok=True)


@app.get("/")
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        return Response(
            f"Training complete. Test F1: {artifact.test_metric.f1_score:.4f}"
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("models/preprocessor.pkl")
        model = load_object("models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        df["predicted_column"] = network_model.predict(df)
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
