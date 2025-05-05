from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_score

app = FastAPI(title="GrammaticaX Essay Scoring API", debug=True)


class EssayInput(BaseModel):
    essay: str

@app.post("/predict")
def get_score(input: EssayInput):
    result = predict_score(input.essay)
    return result

