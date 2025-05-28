from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from model import AmongUsModel
from agents import Imposter
from call_label_agent import CellLabelAgent # Make sure to import CellLabelAgent
from voting import VotingDisplay
import socket
import time
from mesa.visualization.UserParam import UserSettableParameter

def agent_portrayal(agent):
    if isinstance(agent, CellLabelAgent):
        return {
            "Shape": "rect",
            "Color": "#f0f0f0",
            "Filled": "true",
            "Layer": 0,
            "w": len(str(agent.label)) * 0.6,  # Adjust width based on text length
            "h": 1,
            "text": str(agent.label),
            "text_color": "black",
            "text_size": 12
        }
    elif isinstance(agent, Imposter):
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 1,
            "Color": "red",
            "r": 0.5 if agent.alive else 0.2,
            "text": str(agent.unique_id),
            "text_color": "white"
        }
    else:
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 1,
            "Color": "blue",
            "r": 0.5 if agent.alive else 0.2,
            "text": str(agent.unique_id),
            "text_color": "white"
        }

def agent_portrayal_with_rooms(agent):
    if agent is None:
        # Draw rooms as background
        return []
    return agent_portrayal(agent)

def draw_rooms(model):
    portrayal = []
    for room in model.rooms:
        portrayal.append({
            "Shape": "rect",
            "Color": "#f0f0f0",
            "Filled": "true",
            "Layer": 0,
            "x": room[0],
            "y": room[1],
            "w": room[2] - room[0] + 1,
            "h": room[3] - room[1] + 1,
            "text": room[4],
            "text_color": "black"
        })
    return portrayal

def draw_rooms_on_grid(canvas, grid):
    # This function is not needed with the wrapper approach
    pass

def find_free_port():
    """Find a free port using temporary socket"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_server():
    port = find_free_port()
    print(f"Starting server on port {port}")

    grid = CanvasGrid(agent_portrayal_with_rooms, 20, 20, 500, 500)
    voting_display = VotingDisplay()

    # Use UserSettableParameter for interactive model parameters
    model_params = {
        "num_agents": UserSettableParameter('number', 'Number of Crewmates', 4),
        "num_imposters": UserSettableParameter('number', 'Number of Imposters', 1),
        "width": 20,
        "height": 20
    }

    server = ModularServer(
        AmongUsModel,
        [grid, voting_display],
        "Among Us Simulation",
        model_params,
        port=port
    )

    try:
        server.launch()
    except OSError as e:
        print(f"Port error: {e}")
        print("Retrying with new port in 2 seconds...")
        time.sleep(2)
        run_server()
    except KeyboardInterrupt:
        print("\nServer shut down successfully")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_server()