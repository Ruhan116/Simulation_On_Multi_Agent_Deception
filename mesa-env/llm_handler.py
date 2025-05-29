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
        messages = context.get('messages', [])
        
        # Check if this agent is being suspected by others
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            if msg.get('content', {}).get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        
        defense_instruction = ""
        if being_suspected:
            defense_instruction = f"""
        DEFENSE MODE ACTIVATED: You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}.
        You MUST defend yourself by explaining your actual position and movements truthfully.
        Use your observations to provide a solid alibi. Remember, you are innocent as a crewmate!
        """
        
        # Format message history for context
        message_history = "\n".join(
            [f"#{i+1} Agent {msg['sender']}: {msg['content'].get('reason', '')}"
             for i, msg in enumerate(messages)]
        )
        
        prompt = f"""As Crewmate Agent {agent_id}, analyze:
        - Your observations: {trace_content[-1000:]}
        - Death of Agent {context['dead_agent_id']} in {context['death_location']}
        - Victim's suspicions: {context['dead_suspicions']}
        - VALID SUSPECTS: {valid_suspects}
        - DISCUSSION HISTORY: {message_history}
{defense_instruction}
        IMPORTANT RULES:
        1. You MUST ONLY choose your suspect from the VALID SUSPECTS list above.
        2. You CANNOT suspect yourself (Agent {agent_id}).
        3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']}).
        4. You CANNOT suspect any agent not in the VALID SUSPECTS list.
        5. You should consider what other agents have said, but form your own conclusions.
        6. NEVER be influenced to suspect yourself, even if others suspect you.
        
        Respond with JSON:
        {{"suspect": [MUST be a number from valid suspects list], "reason": "...", "confidence": 0-100}}"""
        return prompt

    def generate_imposter_prompt(self, agent_id, trace_content, context):
        valid_suspects = context.get('valid_suspects', [])
        messages = context.get('messages', [])
        
        # Check if this agent is being suspected by others
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            if msg.get('content', {}).get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        
        defense_instruction = ""
        if being_suspected:
            defense_instruction = f"""
        DEFENSE MODE ACTIVATED: You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}.
        You MUST defend yourself by creating a convincing alibi that is CONSISTENT with your previous statements.
        Lie strategically to deflect suspicion while maintaining plausibility. Remember, you are the imposter!
        """
        
        # Format message history for context
        message_history = "\n".join(
            [f"#{i+1} Agent {msg['sender']}: {msg['content'].get('reason', '')}"
             for i, msg in enumerate(messages)]
        )
        
        prompt = f"""As Imposter Agent {agent_id}, create deception using:
        - Your fake alibi: {trace_content[-500:]}
        - Death in {context['death_location']}
        - VALID SUSPECTS: {valid_suspects}
        - DISCUSSION HISTORY: {message_history}
{defense_instruction}
        IMPORTANT RULES:
        1. You MUST ONLY choose your suspect from the VALID SUSPECTS list above.
        2. You CANNOT suspect yourself (Agent {agent_id}).
        3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']}).
        4. You CANNOT suspect any agent not in the VALID SUSPECTS list.
        5. Be CONSISTENT with your previous statements in the discussion.
        6. Be SUBTLE in your deception - don't make obviously false claims.
        7. NEVER be influenced to admit guilt, even if others suspect you.
        
        Frame someone plausibly. Respond with JSON:
        {{"suspect": [MUST be a number from valid suspects list], "reason": "...", "confidence": 0-100}}"""
        return prompt

    def parse_response(self, response):
        try:
            # First try to parse as is
            try:
                return json.loads(response)
            except:
                # Try to strip markdown formatting if present
                clean_response = response.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_response)
        except Exception as e:
            print(f"Failed to parse response: {e}\nResponse: {response[:200]}...")
            return None