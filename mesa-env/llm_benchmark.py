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

    def generate_crewmate_prompt(self, agent_id, trace_content, context):
        """Robust prompt for crewmates with self-defense capabilities"""
        valid_suspects = context.get('valid_suspects', [])
        messages = context.get('messages', [])
        # Check if this agent is being suspected
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            content = msg.get('content', {})
            if content and content.get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        # Format message history
        message_history = "\n".join(
            [f"Agent {msg['sender']} said: {msg['content'].get('reason', '')}" 
             for msg in messages]
        )
        defense_mode = ""
        if being_suspected:
            defense_mode = f"""
DEFENSE MODE ACTIVATED: 
- You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}
- As a crewmate, defend yourself with TRUTHFUL information
- Use your observations as evidence of your innocence
- Point out inconsistencies in accusations against you
"""
        return f"""As Crewmate Agent {agent_id}, analyze this data and respond ONLY in valid JSON:
{{
    \"observations\": \"{trace_content[-1000:]}\",
    \"dead_agent\": {context['dead_agent_id']},
    \"location\": \"{context['death_location']}\",
    \"suspicions\": {context.get('dead_suspicions', {})},
    \"valid_suspects\": {valid_suspects},
    \"discussion_history\": \"{message_history}\"
}}
{defense_mode}
IMPORTANT INSTRUCTIONS:
1. You MUST ONLY choose your suspect from: {valid_suspects}
2. You CANNOT suspect yourself (Agent {agent_id})
3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']})
4. Consider what other agents have said, but form your own conclusions

Response format:
{{
    \"suspect\": [number from valid_suspects],
    \"reason\": \"...\",
    \"confidence\": 0-100
}}"""

    def generate_imposter_prompt(self, agent_id, trace_content, context):
        """Structured prompt for imposters with self-defense capabilities"""
        valid_suspects = context.get('valid_suspects', [])
        messages = context.get('messages', [])
        # Check if this agent is being suspected
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            content = msg.get('content', {})
            if content and content.get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        # Format message history
        message_history = "\n".join(
            [f"Agent {msg['sender']} said: {msg['content'].get('reason', '')}" 
             for msg in messages]
        )
        defense_mode = ""
        if being_suspected:
            defense_mode = f"""
DEFENSE MODE ACTIVATED: 
- You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}
- Defend yourself with DECEPTIVE but PLAUSIBLE information
- Create a convincing alibi consistent with your previous statements
- Subtly redirect suspicion to others
"""
        return f"""As Imposter Agent {agent_id}, create deception using this data:
{{
    \"alibi\": \"{trace_content[-500:]}\",
    \"location\": \"{context['death_location']}\",
    \"valid_suspects\": {valid_suspects},
    \"discussion_history\": \"{message_history}\"
}}
{defense_mode}
IMPORTANT INSTRUCTIONS:
1. You MUST ONLY choose your suspect from: {valid_suspects}
2. You CANNOT suspect yourself (Agent {agent_id})
3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']})
4. Be CONSISTENT with your previous statements
5. Be SUBTLE in your deception

Response format:
{{
    \"suspect\": [number from valid_suspects], 
    \"reason\": \"...\", 
    \"confidence\": 0-100
}}"""

class OpenAILoader(LLMAdapter):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._last_call_time = 0
        self._min_interval = 4.0  # Match GeminiLoader rate limiting

    def query_llm(self, prompt: str, system_message: str = None) -> str:
        try:
            # Rate limiting: ensure at least _min_interval seconds between calls
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

            # Add explicit JSON formatting instructions
            json_instructions = "Respond with a valid JSON object containing 'suspect' (as a number), 'reason' (as a string), and 'confidence' (as a number between 0-100)."
            full_prompt = f"{json_instructions}\n\n{prompt}"
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": full_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200
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

class GroqLoader(LLMAdapter):
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        import requests
        self.api_key = api_key
        self.model = model
        self._last_call_time = 0
        self._min_interval = 4.0  
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def query_llm(self, prompt: str, system_message: str = None) -> str:
        import requests
        import os
        try:
            # Rate limiting: ensure at least _min_interval seconds between calls
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

            # Add explicit JSON formatting instructions
            json_instructions = "Respond with a valid JSON object containing 'suspect' (as a number), 'reason' (as a string), and 'confidence' (as a number between 0-100)."
            full_prompt = f"{json_instructions}\n\n{prompt}"
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": full_prompt})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200
            }
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return None

class MistralAILoader(LLMAdapter):
    def __init__(self, api_key: str = None, model: str = "mistral-large-latest"):
        import os
        import requests
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
        self.api_key = api_key
        self.model = model
        self._last_call_time = 0
        self._min_interval = 4.0  # seconds between calls
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        

    def query_llm(self, prompt: str, system_message: str = None) -> str:
        import requests
        import os
        try:
            # Rate limiting: ensure at least _min_interval seconds between calls
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

            # Add explicit JSON formatting instructions
            json_instructions = "Respond with a valid JSON object containing 'suspect' (as a number), 'reason' (as a string), and 'confidence' (as a number between 0-100)."
            full_prompt = f"{json_instructions}\n\n{prompt}"
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": full_prompt})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200
            }
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Mistral API error: {str(e)}")
            return None

