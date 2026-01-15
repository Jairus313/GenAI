import warnings

warnings.filterwarnings("ignore")

import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.rag_service import RAGService

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


class LLM:
    def generate(self, prompt: str) -> str:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        response = model.generate_content(
            prompt, generation_config={"temperature": 0.2, "max_output_tokens": 100000}
        )

        try:
            return response.text
        except ValueError:
            return "Something went wrong, can please try again"


def load_chunks():
    with open("data/menu.txt") as f:
        text = f.read()

    # naive chunking for now
    return [text[i : i + 500] for i in range(0, len(text), 500)]


# Setup
chunks = load_chunks()
embedder = Embedder()
embeddings = embedder.embed(chunks)

store = VectorStore(dim=len(embeddings[0]))
store.add(embeddings, chunks)

rag = RAGService(embedder, store, LLM())


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest):
    answer = rag.answer(payload.question)
    return {"answer": answer}
