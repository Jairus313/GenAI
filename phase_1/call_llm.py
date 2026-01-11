# test_gemini.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

response = model.generate_content(
    contents="explain llms as if I am five",
)

print(f"\n\nGemini Response:\n{response.text}")
