import json
import requests
from openai import OpenAI

class OpenAIHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def query_llm(self, prompt, system_message=None):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

class DiscussionManager:
    def __init__(self, api_key):
        self.llm = OpenAIHandler(api_key)
    
    def generate_crewmate_prompt(self, agent_id, trace_content, context):
        valid_suspects = context.get('valid_suspects', [])
        suspects_str = ', '.join(str(s) for s in valid_suspects)
        prompt = f"""
AS CREWMATE AGENT {agent_id}:
- Your observations: {trace_content[-1000:]}
- Death of Agent {context['dead_agent_id']} in {context['death_location']}
- Victim's suspicions: {context['dead_suspicions']}

***VALID SUSPECTS (CHOOSE ONLY FROM THIS LIST): [{suspects_str}]***

IMPORTANT: YOU ARE FORBIDDEN TO SUSPECT YOURSELF, DEAD, OR OUT-OF-RANGE AGENTS. YOU MUST PICK EXACTLY ONE SUSPECT FROM THE ABOVE LIST.

Respond ONLY with JSON in this format:
{{"suspect": [number from valid suspects], "reason": "...", "confidence": 0-100}}
"""
        return prompt

    def generate_imposter_prompt(self, agent_id, trace_content, context):
        valid_suspects = context.get('valid_suspects', [])
        suspects_str = ', '.join(str(s) for s in valid_suspects)
        prompt = f"""
AS IMPOSTER AGENT {agent_id} (deceive and frame):
- Your fake alibi: {trace_content[-500:]}
- Death in {context['death_location']}

***VALID SUSPECTS (CHOOSE ONLY FROM THIS LIST): [{suspects_str}]***

IMPORTANT: YOU ARE FORBIDDEN TO SUSPECT YOURSELF, DEAD, OR OUT-OF-RANGE AGENTS. YOU MUST PICK EXACTLY ONE SUSPECT FROM THE ABOVE LIST.

Respond ONLY with JSON in this format:
{{"suspect": [number from valid suspects], "reason": "...", "confidence": 0-100}}
"""
        return prompt

    def parse_response(self, response):
        try:
            return json.loads(response.strip("```json\n").rstrip("\n```"))
        except:
            return None