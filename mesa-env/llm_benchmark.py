from abc import ABC, abstractmethod
import json
import re
import random
import time

class LLMAdapter(ABC):
    @abstractmethod
    def query_llm(self, prompt: str, system_message: str = None) -> str:
        pass

    @staticmethod
    def parse_response(response: str) -> dict:
        try:
            # Handle Gemini's weird array responses
            if response.startswith('['):
                first_item = json.loads(response)[0]
                return {
                    "suspect": first_item.get("suspect", -1),
                    "reason": first_item.get("reason", ""),
                    "confidence": first_item.get("confidence", 50)
                }
                
            # Normal JSON parsing with markdown cleanup
            clean = re.sub(r'^```json|```$', '', response, flags=re.MULTILINE).strip()
            parsed = json.loads(clean)
            return {
                "suspect": parsed.get("suspect", -1),
                "reason": parsed.get("reason", ""),
                "confidence": parsed.get("confidence", 50)
            }
        except Exception as e:
            print(f"Final fallback parsing for: {response}")
            # Robust regex extraction
            suspect = re.findall(r'\b\d+\b', response)
            return {
                "suspect": int(suspect[0]) if suspect else -1,
                "reason": "Automatic parse",
                "confidence": 50
            }

class OpenAILoader(LLMAdapter):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def query_llm(self, prompt: str, system_message: str = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return None

class GeminiLoader(LLMAdapter):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self._last_call_time = 0
        self._min_interval = 4.0  # Increased seconds between calls

    def query_llm(self, prompt: str, system_message: str = None) -> str:
        try:
            # Rate limiting: ensure at least _min_interval seconds between calls
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            # Add explicit JSON formatting instructions
            json_instructions = "Respond with a valid JSON object containing 'suspect' (as a number), 'reason' (as a string), and 'confidence' (as a number between 0-100)."
            response = self.model.generate_content(
                f"{json_instructions}\n\n{full_prompt}",
                generation_config={"temperature": 0.7, "max_output_tokens": 200}
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return None