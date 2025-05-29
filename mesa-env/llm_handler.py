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
        prompt = f"""As Crewmate Agent {agent_id}, analyze:
        - Your observations: {trace_content[-1000:]}
        - Death of Agent {context['dead_agent_id']} in {context['death_location']}
        - Victim's suspicions: {context['dead_suspicions']}
        - VALID SUSPECTS: {valid_suspects}

        You MUST ONLY choose your suspect from the VALID SUSPECTS list. Do NOT suspect yourself, dead, or out-of-range agents.
        Respond with JSON:
        {{"suspect": [number from valid suspects], "reason": "...", "confidence": 0-100}}"""
        return prompt

    def generate_imposter_prompt(self, agent_id, trace_content, context):
        valid_suspects = context.get('valid_suspects', [])
        prompt = f"""As Imposter Agent {agent_id}, create deception using:
        - Your fake alibi: {trace_content[-500:]}
        - Death in {context['death_location']}
        - VALID SUSPECTS: {valid_suspects}

        You MUST ONLY choose your suspect from the VALID SUSPECTS list. Do NOT suspect yourself, dead, or out-of-range agents.
        Frame someone plausibly. Respond with JSON:
        {{"suspect": [number from valid suspects], "reason": "...", "confidence": 0-100}}"""
        return prompt

    def parse_response(self, response):
        try:
            return json.loads(response.strip("```json\n").rstrip("\n```"))
        except:
            return None