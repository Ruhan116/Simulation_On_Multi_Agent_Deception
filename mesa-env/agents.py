from mesa import Model, Agent
from task import Task
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random



class PlayerAgent(Agent):
    def __init__(self, unique_id, model, visibility):
        super().__init__(unique_id, model)
        self.visibility = visibility
        self.alive = True

    def move_toward(self, target_location):
        """Move 1 cell toward target location only if valid and not already there."""
        if not target_location or self.pos == target_location:
            return

        x, y = self.pos
        tx, ty = target_location

        dx = tx - x
        dy = ty - y

        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0

        # Try x-axis first
        if step_x != 0:
            new_pos = (x + step_x, y)
            if self.model.is_valid_position(new_pos):
                self.model.grid.move_agent(self, new_pos)
                return

        # Then try y-axis
        if step_y != 0:
            new_pos = (x, y + step_y)
            if self.model.is_valid_position(new_pos):
                self.model.grid.move_agent(self, new_pos)
                return

        # Finally try diagonal
        if step_x != 0 and step_y != 0:
            new_pos = (x + step_x, y + step_y)
            if self.model.is_valid_position(new_pos):
                self.model.grid.move_agent(self, new_pos)


class Crewmate(PlayerAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, visibility=6)
        main_rooms = [room for room in model.rooms if room[4] != "Hallway"]
        self.tasks = [
            Task(f"{room[4]} Task", ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2))
            for room in main_rooms
        ]
        self.suspicion_pairs = {}  # Format: {frozenset: {"count": int, "rooms": list}}

    def update_suspicions(self, visible_agents):
        current_room = self.model.get_room(self.pos)
        visible_players = [
            a for a in visible_agents 
            if a != self and isinstance(a, (Crewmate, Imposter))
        ]
        
        # Initialize suspicion_pairs if missing
        if not hasattr(self, 'suspicion_pairs'):
            self.suspicion_pairs = {}

        # Collect all unique pairs this step for tracing
        trace_pairs = set()
        # Track pairs for suspicion (separate from tracing)
        for i in range(len(visible_players)):
            for j in range(i + 1, len(visible_players)):
                agent1 = visible_players[i].unique_id
                agent2 = visible_players[j].unique_id
                pair = frozenset({agent1, agent2})

                # For tracing (your original format)
                trace_pairs.add(f"{{Agent {min(agent1, agent2)}, Agent {max(agent1, agent2)}, {current_room}}}")

                # Initialize pair data structure properly
                if pair not in self.suspicion_pairs:
                    self.suspicion_pairs[pair] = {"count": 0, "rooms": []}
                
                # Update count (not the whole dict)
                self.suspicion_pairs[pair]["count"] += 1
                self.suspicion_pairs[pair]["rooms"].append(current_room)
        
        # Write to trace file
        if not hasattr(self, '_trace_file'):
            self._trace_file = open(f"agent_{self.unique_id}_trace.log", "w")
        self._trace_file.write(
            f"Step {self.model.schedule.steps}: [{', '.join(sorted(trace_pairs))}]\n"
        )

        trace_line = f"Step {self.model.schedule.steps}: "
        trace_line += f"Alive({self.alive}), "
        trace_line += f"Pos({self.pos}), "
        trace_line += f"Visible: {[a.unique_id for a in visible_agents if a != self]}"
        
        self._trace_file.write(trace_line + "\n")
        self._trace_file.flush()
        
        # Debug output
        # print(f"Agent {self.unique_id} suspicion_pairs: {self.suspicion_pairs}")

        
    def close_trace_file(self):
        if hasattr(self, '_trace_file'):
            self._trace_file.close()

    def find_nearest_task(self):
        closest, min_dist = None, float("inf")
        x, y = self.pos
        for task in self.tasks:
            if task.complete:
                continue
            tx, ty = task.location
            dist = abs(tx - x) + abs(ty - y)
            if dist < min_dist:
                closest, min_dist = task, dist
        return closest

    def do_task(self, task):
        if self.pos == task.location:
            task.do_task()
            if task.complete:
                print(f"Agent {self.unique_id} completed {task.name}!")
                
                # Check if this agent has completed all their tasks
                if all(t.complete for t in self.tasks):
                    print(f"\nAgent {self.unique_id} has completed all their tasks!")
                    # List remaining agents with incomplete tasks
                    agents_with_tasks = [
                        a for a in self.model.schedule.agents
                        if isinstance(a, Crewmate) and a.alive and not all(t.complete for t in a.tasks)
                    ]
                    if agents_with_tasks:
                        print("Agents still with incomplete tasks:")
                        for agent in agents_with_tasks:
                            incomplete = sum(1 for t in agent.tasks if not t.complete)
                            print(f"- Agent {agent.unique_id}: {incomplete} tasks remaining")
                    else:
                        print("All alive crewmates have completed their tasks!")

    
    def check_for_bodies(self, visible_agents):
        if self.model.phase != "tasks":
            return False
        for agent in visible_agents:
            if isinstance(agent, (Crewmate, Imposter)) and not agent.alive:
                print(f"Agent {self.unique_id} found body of {agent.unique_id}!")
                self.model.reported_body = agent.pos
                self.model.phase = "discussion"
                return True
        return False
        
    def generate_argument(self, discussion_manager, context):
        try:
            with open(f"agent_{self.unique_id}_trace.log", "r") as f:
                trace_content = f.read()
        except FileNotFoundError:
            trace_content = ""
            
        prompt = discussion_manager.generate_crewmate_prompt(
            self.unique_id, 
            trace_content,
            context
        )
        response = discussion_manager.llm.query_llm(prompt)
        return discussion_manager.parse_response(response)


    def get_dead_agent_pairs(self, dead_id):
        """Extract suspicion pairs involving dead agent"""
        return {
            pair: data for pair, data in self.suspicion_pairs.items()
            if dead_id in pair
        }
    
    def calculate_heuristic_suspicion(self, suspect_id):
        """Calculate suspicion based on observed pairs"""
        return sum(
            data["count"] 
            for pair, data in self.suspicion_pairs.items()
            if suspect_id in pair
        )

    def step(self):
        if not self.alive:
            return

        task = self.find_nearest_task()
        if task:
            self.move_toward(task.location)
            self.do_task(task)

        visible_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.visibility, include_center=True
        )

        self.update_suspicions(visible_agents)
        self.check_for_bodies(visible_agents)


class Imposter(PlayerAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, visibility=9)
        self.kill_cooldown = 0
        self.patrol_index = 0
        # Define patrol route (center of each room in order)
        self.patrol_route = [
            ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2)
            for room in model.rooms
        ]
        self.current_target = None
        self.stalking_target = None
        main_rooms = [room for room in model.rooms if room[4] != "Hallway"]
        self.fake_tasks = [
            Task(f"Fake {room[4]} Task", ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2))
            for room in main_rooms
        ]

    def find_vulnerable_target(self):
        """Find best kill target based on isolation and proximity"""
        visible_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.visibility, include_center=True
        )
        # Filter to alive crewmates
        potential_targets = [
            a for a in visible_agents 
            if isinstance(a, Crewmate) and a.alive
        ]
        # Find most isolated target
        best_target = None
        min_nearby = float('inf')
        for target in potential_targets:
            # Check nearby agents (radius=1)
            nearby = self.model.grid.get_neighbors(
                target.pos, moore=True, radius=1
            )
            nearby_alive = [a for a in nearby if a.alive and a != self]
            # Prioritize targets with fewest nearby agents
            if len(nearby_alive) < min_nearby:
                min_nearby = len(nearby_alive)
                best_target = target
        return best_target if min_nearby <= 1 else None

    def is_adjacent(self, agent):
        """Check if agent is directly adjacent (including diagonals)"""
        dx = abs(self.pos[0] - agent.pos[0])
        dy = abs(self.pos[1] - agent.pos[1])
        return dx <= 1 and dy <= 1 and (dx != 0 or dy != 0)
    
    def kill(self, target):
        if target.alive and self.is_adjacent(target):
            target.alive = False
            self.kill_cooldown = 5
            print(f"Agent {target.unique_id} was killed!")
            self.stalking_target = None  # Reset stalking after kill

    def step(self):
        if not self.alive or self.kill_cooldown > 0:
            if self.kill_cooldown > 0:
                self.kill_cooldown -= 1
            return
        # Try to find a vulnerable target to kill
        target = self.find_vulnerable_target()
        if target:
            # If we're close enough, kill immediately
            if self.is_adjacent(target):
                self.kill(target)
                return
            # Otherwise start stalking
            self.stalking_target = target
            self.move_toward(target.pos)
            return
        # Reset stalking if no target
        self.stalking_target = None
        # If we see multiple agents, blend in by doing fake tasks
        visible_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.visibility, include_center=True
        )
        visible_crewmates = [
            a for a in visible_agents 
            if isinstance(a, Crewmate) and a.alive
        ]
        if len(visible_crewmates) >= 2:
            # Move to nearest fake task location
            closest_task = None
            min_dist = float('inf')
            for room in self.model.rooms[:4]:  # Only main rooms
                task_pos = ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2)
                dist = abs(self.pos[0] - task_pos[0]) + abs(self.pos[1] - task_pos[1])
                if dist < min_dist:
                    closest_task = task_pos
                    min_dist = dist
            if closest_task:
                self.move_toward(closest_task)
            return
        # Default behavior: patrol rooms
        target_pos = self.patrol_route[self.patrol_index]
        # Move toward current patrol point
        self.move_toward(target_pos)
        # If reached patrol point, go to next
        if self.pos == target_pos:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_route)
