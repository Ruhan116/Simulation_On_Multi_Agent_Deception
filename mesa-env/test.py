import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get your API key
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

# Configure the Gemini client
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Test the key by making a simple request
try:
    response = model.generate_content(
        "What was his deadlift and bench press max?",
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=150,
        )
    )
    print("Gemini API response:", response.text)
except Exception as e:
    print("An error occurred:", str(e))