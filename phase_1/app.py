import warnings

warnings.filterwarnings("ignore")

import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel

from llm_service import GeminiService

load_dotenv()

app = FastAPI()

llm = GeminiService(api_key=os.getenv("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    try:
        answer = llm.generate(payload.question)
        return AskResponse(answer=answer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM service failed. Try again later.\n{e}",
        )
