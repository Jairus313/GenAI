import warnings

warnings.filterwarnings("ignore")

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # low randomness
                "max_output_tokens": 100000,  # max tokens
            },
        )
        return response.text


if __name__ == "__main__":
    gemini_client = GeminiService(os.getenv("GEMINI_API_KEY"))

    prompt = """
            teach about the importance of LLMs and why professional needs to be learning it
            in this year with their jobs, explain it in reason, need and conclusion but 
            keep it short and concise.
    """

    gemini_response = gemini_client.generate(prompt)

    print(f"\n\nGemini Response:\n{gemini_response}")
