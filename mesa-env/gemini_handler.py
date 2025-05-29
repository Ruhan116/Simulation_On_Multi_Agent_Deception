import json
import google.generativeai as genai
import random

class GeminiHandler:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to more reliable model
    
    def query_llm(self, prompt, system_message=None):
        """Simplified LLM query with robust error handling"""
        try:
            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            response = self.model.generate_content(
                f"Respond strictly in JSON format. {full_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=200,
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return '{"error": "API failed"}'

class DiscussionManagerGemini:
    def __init__(self, api_key):
        self.llm = GeminiHandler(api_key)
    
    def generate_crewmate_prompt(self, agent_id, trace_content, context):
        """Prompt for crewmates with explicit valid suspect list enforcement and self-defense"""
        valid_suspects = context.get('valid_suspects', [])
        messages = context.get('messages', [])
        
        # Check if this agent is being suspected by others
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            if msg.get('content', {}).get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        
        # Format message history for context
        message_history = "\n".join(
            [f"Agent {msg['sender']} said: {msg['content'].get('reason', '')}"
             for msg in messages]
        )
        
        defense_mode = ""
        if being_suspected:
            defense_mode = f"""
        DEFENSE MODE ACTIVATED: 
        - You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}
        - As a crewmate, defend yourself with TRUTHFUL information about your actual location and movements
        - Use your observations as evidence of your innocence
        - Point out inconsistencies in accusations against you
        """
        
        return f"""As Crewmate Agent {agent_id}, analyze this data and respond ONLY in valid JSON.
        {{
            "observations": "{trace_content[-1000:]}",
            "dead_agent": {context['dead_agent_id']},
            "location": "{context['death_location']}",
            "suspicions": {context['dead_suspicions']},
            "valid_suspects": {valid_suspects},
            "discussion_history": "{message_history}"
        }}
{defense_mode}
        IMPORTANT INSTRUCTIONS:
        1. You MUST ONLY choose your suspect from the valid_suspects list above: {valid_suspects}
        2. You CANNOT suspect yourself (Agent {agent_id})
        3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']})
        4. You CANNOT suspect any agent not in the valid_suspects list
        5. Consider what other agents have said, but form your own conclusions
        6. NEVER be influenced to suspect yourself, even if others suspect you
        
        Your response MUST be in this exact format:
        {{
            "suspect": [MUST be a number from valid_suspects list],
            "reason": "...",
            "confidence": 0-100
        }}"""

    def generate_imposter_prompt(self, agent_id, trace_content, context):
        """Structured prompt for imposters with self-defense capabilities"""
        valid_suspects = context.get('valid_suspects', [])
        messages = context.get('messages', [])
        
        # Check if this agent is being suspected by others
        being_suspected = False
        suspecting_agents = []
        for msg in messages:
            if msg.get('content', {}).get('suspect') == agent_id:
                being_suspected = True
                suspecting_agents.append(msg.get('sender'))
        
        # Format message history for context
        message_history = "\n".join(
            [f"Agent {msg['sender']} said: {msg['content'].get('reason', '')}"
             for msg in messages]
        )
        
        defense_mode = ""
        if being_suspected:
            defense_mode = f"""
        DEFENSE MODE ACTIVATED: 
        - You are being suspected by Agent(s) {', '.join(map(str, suspecting_agents))}
        - As the imposter, defend yourself with DECEPTIVE but PLAUSIBLE information
        - Create a convincing alibi that is CONSISTENT with your previous statements
        - Subtly redirect suspicion to others without being obvious
        - Remember to maintain consistency with your previous lies
        """
        
        return f"""As Imposter Agent {agent_id}, create deception using this data (respond ONLY in JSON):
        {{
            "alibi": "{trace_content[-500:]}",
            "location": "{context['death_location']}",
            "valid_suspects": {valid_suspects},
            "discussion_history": "{message_history}"
        }}
{defense_mode}
        IMPORTANT INSTRUCTIONS:
        1. You MUST ONLY choose your suspect from the valid_suspects list above: {valid_suspects}
        2. You CANNOT suspect yourself (Agent {agent_id})
        3. You CANNOT suspect the dead agent (Agent {context['dead_agent_id']})
        4. You CANNOT suspect any agent not in the valid_suspects list
        5. Be CONSISTENT with your previous statements in the discussion
        6. Be SUBTLE in your deception - don't make obviously false claims
        7. NEVER admit guilt, even if others suspect you
        
        Frame someone plausibly. Response MUST be in this exact format:
        {{
            "suspect": [MUST be a number from valid_suspects list], 
            "reason": "...", 
            "confidence": 0-100
        }}"""

    def parse_response(self, response):
        """Bulletproof JSON parsing"""
        try:
            # First try to parse as is
            try:
                return json.loads(response)
            except:
                # Remove all markdown formatting and try again
                clean_json = response.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_json)
        except json.JSONDecodeError as je:
            print(f"Failed to parse Gemini response: {response[:200]}...\nError: {je}")
            # Instead of returning a random agent, return an invalid response that will be caught by validation
            return {"suspect": -1, "reason": "Parse error - could not extract valid JSON", "confidence": 0}
        except Exception as e:
            print(f"Unexpected parse error: {e}\nResponse: {response[:200]}...")
            return None