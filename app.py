from scripts import models,s3
from fastapi import FastAPI
import os
from fastapi import Request
from scripts.models import *
import uvicorn
import torch
from transformers import pipeline
import time

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

app = FastAPI()

##### Download ML Models #####

model_name = 'tinyBERT-sentiment-analysis/'
local_path = 'ml_models/'+model_name

if not os.path.isdir(local_path):
    s3.download_dir(local_path,model_name)

sentiment_model = pipeline('text-classification',model = local_path,device = device)

##### Download ENDS #####

# Load Model : 


@app.get("/")
def read_root():
    return "Hello! I am Alive "

@app.post("/api/v1/get_sentiment")
def sentiment_analysis(data : NLPDataInput):
    # print("✅ Parsed request:", data.dict())
    # return {"received": data.dict()}
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = end-start

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-sentiment_analysis",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    return output

if __name__ == '__main__':
    uvicorn.run(app="app:app",reload=True)