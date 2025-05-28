from mesa.visualization.modules import TextElement
from agents import Imposter  # Add this import at the top

class VotingDisplay(TextElement):
    def __init__(self):
        super().__init__()
    
    def render(self, model):
        if model.phase == "voting":
            # Show last 5 messages
            recent_msgs = "<br>".join(
                [f"Agent {msg['sender']}: {msg['content'].get('reason', '')}" 
                 for msg in model.message_queue[-5:]])
            # Show voting results as before
            votes = []
            for agent_id, vote_count in model.votes.items():
                agent = next((a for a in model.schedule.agents if a.unique_id == agent_id), None)
                if agent:
                    color = "red" if isinstance(agent, Imposter) else "blue"
                    votes.append(f'<span style="color:{color}">Agent {agent_id}: {vote_count} votes</span>')
            vote_html = "<h3>Voting Results:</h3><ul>" + "".join([f"<li>{v}</li>" for v in votes]) + "</ul>"
            return f"<h3>Recent Arguments:</h3><div>{recent_msgs}</div>" + vote_html
        elif model.phase == "discussion":
            return "<h3>Discussion Phase</h3><p>Discussing who the imposter might be...</p>"
        else:
            return "<h3>Task Phase</h3><p>Crewmates completing tasks, Imposters lurking...</p>"
