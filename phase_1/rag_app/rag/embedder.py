import google.generativeai as genai


class Embedder:
    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        for text in texts:
            response = genai.embed_content(model=self.model_name, content=text)
            embeddings.append(response["embedding"])

        return embeddings
