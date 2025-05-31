from abc import ABC, abstractmethod
import json
import re
import random
import time

class LLMAdapter(ABC):
    @abstractmethod
    def query_llm(self, prompt: str, system_message: str = None) -> str:
        pass

    VALID_STATEMENT_TYPES = {
        'accusation', 'defense', 'alibi', 'observation', 'vote_request'
    }

    @staticmethod
    def parse_response(response: str) -> dict:
        def _infer_fallback_type(text: str) -> str:
            text = text.lower()
            if any(word in text for word in ['vote', 'eject', 'kick out']):
                return 'vote_request'
            elif any(word in text for word in ['saw', 'notice', 'observed']):
                return 'observation'
            elif any(word in text for word in ['with me', 'together', 'alibi']):
                return 'alibi'
            elif any(word in text for word in ['defend', 'innocent', 'wrong']):
                return 'defense'
            return 'accusation'

        try:
            # Remove markdown code block markers if present
            response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE).strip()
            # Try to extract JSON object from the response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                response = match.group(0)
            # Handle Gemini's weird array responses
            if response.startswith('['):
                first_item = json.loads(response)[0]
                parsed = {
                    "suspect": first_item.get("suspect", -1),
                    "reason": first_item.get("reason", ""),
                    "confidence": first_item.get("confidence", 50),
                    "statement_type": first_item.get("statement_type", "observation")
                }
            else:
                parsed_json = json.loads(response)
                parsed = {
                    "suspect": parsed_json.get("suspect", -1),
                    "reason": parsed_json.get("reason", ""),
                    "confidence": parsed_json.get("confidence", 50),
                    "statement_type": parsed_json.get("statement_type", "observation")
                }
            # Validate statement_type
            stype = str(parsed.get("statement_type", "observation")).lower()
            if stype not in LLMAdapter.VALID_STATEMENT_TYPES:
                stype = _infer_fallback_type(parsed.get("reason", ""))
            parsed["statement_type"] = stype
            return parsed
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {
                "suspect": -1,
                "reason": "(Invalid or unparsable LLM response)",
                "confidence": 0,
                "statement_type": "observation"
            }

    GENERAL_RULES = """
GENERAL RULES:
1. You may ONLY reference:
   - Agents you've actually seen (from Visible: lists)
   - Rooms you've actually been in
   - Pairs of agents you've seen together
2. Our ship only has these rooms: Cafeteria, Weapons, Navigation, Shields, Hallway
3. Never mention non-existent rooms or systems
4. Your statements must be directly supported by your trace log
"""

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
        return f"""
As Crewmate Agent {agent_id}, analyze this data and respond in STRICT JSON format:
{{
    "suspect": [number from valid_suspects],
    "reason": "...",
    "confidence": 0-100,
    "statement_type": "accusation|defense|alibi|observation|vote_request"
}}

RULES FOR STATEMENT TYPE:
- Use "accusation" when directly accusing another agent
- Use "defense" when defending yourself or others
- Use "alibi" when providing your whereabouts
- Use "observation" when stating neutral facts
- Use "vote_request" when explicitly asking others to vote
}}
{defense_mode}
IMPORTANT INSTRUCTIONS:
1. You MUST ONLY choose your suspect from: {valid_suspects}
2. You CANNOT suspect yourself (Agent {agent_id})
3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']})
4. Consider what other agents have said, but form your own conclusions

Response format:
{{
    "suspect": [number from valid_suspects],
    "reason": "...",
    "confidence": 0-100,
    "statement_type": "accusation|defense|alibi|observation|vote_request"
}}

RULES FOR STATEMENT TYPE:
- Use "accusation" when directly accusing another agent
- Use "defense" when defending yourself or others
- Use "alibi" when providing your whereabouts
- Use "observation" when stating neutral facts
- Use "vote_request" when explicitly asking others to vote
"""
        # Add general rules to the prompt
        + self.GENERAL_RULES

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
        return f"""
As Imposter Agent {agent_id}, create deception using this data:
{{
    "suspect": [number from valid_suspects], 
    "reason": "...", 
    "confidence": 0-100,
    "statement_type": "accusation|defense|alibi|observation|vote_request"
}}

RULES FOR STATEMENT TYPE:
- Use "accusation" when directly accusing another agent
- Use "defense" when defending yourself or others
- Use "alibi" when providing your whereabouts
- Use "observation" when stating neutral facts
- Use "vote_request" when explicitly asking others to vote
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
    "suspect": [number from valid_suspects], 
    "reason": "...", 
    "confidence": 0-100,
    "statement_type": "accusation|defense|alibi|observation|vote_request"
}}

RULES FOR STATEMENT TYPE:
- Use "accusation" when directly accusing another agent
- Use "defense" when defending yourself or others
- Use "alibi" when providing your whereabouts
- Use "observation" when stating neutral facts
- Use "vote_request" when explicitly asking others to vote
"""
        # Add general rules to the prompt
        + self.GENERAL_RULES

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
        self._min_interval = 10.0
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

