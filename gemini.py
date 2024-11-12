import os
import google.generativeai as genai

# Set the API key
GEMINI_API_KEY = "AIzaSyDbzgMJ_9p3lBy1WbiAu5Ynk4-qgxcxT4c"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)