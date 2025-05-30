import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import numpy as np

class TraceAnalyzer:
    """Analyzes agent movement trace logs to verify location claims and detect deception"""
    
    def __init__(self):
        self.room_mappings = {
            'cafeteria': ['cafeteria', 'cafe', 'eating area'],
            'weapons': ['weapons', 'weapon', 'armory'],
            'navigation': ['navigation', 'nav', 'helm'],
            'shields': ['shields', 'shield', 'defense'],
            'hallway': ['hallway', 'corridor', 'passage'],
            'engine': ['engine', 'reactor', 'power'],
            'security': ['security', 'cameras', 'surveillance'],
            'medbay': ['medbay', 'medical', 'infirmary'],
            'electrical': ['electrical', 'electric', 'wiring'],
            'storage': ['storage', 'supplies', 'warehouse']
        }
        
        # Pattern for parsing trace log entries
        self.trace_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Agent (\d+): (.+)'
        )
        
        # Movement patterns
        self.movement_pattern = re.compile(r'moved to position \((\d+), (\d+)\)')
        self.room_pattern = re.compile(r'entered (.+?) room')
        self.task_pattern = re.compile(r'(started|completed) task (.+?) in (.+)')
        self.interaction_pattern = re.compile(r'interacted with (.+)')
    
    def parse_trace_log(self, trace_content: str, agent_id: int) -> Dict:
        """Parse trace log content and extract structured movement data"""
        if not trace_content.strip():
            return {
                'agent_id': agent_id,
                'movements': [],
                'rooms_visited': [],
                'recent_rooms': [],
                'tasks_performed': [],
                'interactions': [],
                'timeline': [],
                'time_in_rooms': {},
                'movement_patterns': {}
            }
        
        movements = []
        rooms_visited = []
        tasks_performed = []
        interactions = []
        timeline = []
        room_timestamps = defaultdict(list)
        
        lines = trace_content.strip().split('\n')
        
        for line in lines:
            match = self.trace_pattern.match(line)
            if not match:
                continue
            
            timestamp_str, parsed_agent_id, action = match.groups()
            
            # Skip if this isn't the right agent
            if int(parsed_agent_id) != agent_id:
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback timestamp parsing
                timestamp = datetime.now()
            
            # Parse different types of actions
            parsed_action = self._parse_action(action, timestamp)
            if parsed_action:
                timeline.append(parsed_action)
                
                # Track movements
                if parsed_action['type'] == 'movement':
                    movements.append(parsed_action)
                    if 'room' in parsed_action:
                        room_name = parsed_action['room']
                        rooms_visited.append(room_name)
                        room_timestamps[room_name].append(timestamp)
                
                # Track tasks
                elif parsed_action['type'] == 'task':
                    tasks_performed.append(parsed_action)
                    if 'location' in parsed_action:
                        room_timestamps[parsed_action['location']].append(timestamp)
                
                # Track interactions
                elif parsed_action['type'] == 'interaction':
                    interactions.append(parsed_action)
        
        # Calculate time spent in each room
        time_in_rooms = self._calculate_room_durations(room_timestamps)
        
        # Get recent rooms (last 5 unique rooms)
        recent_rooms = list(dict.fromkeys(rooms_visited[-10:]))[-5:]
        
        # Analyze movement patterns
        movement_patterns = self._analyze_movement_patterns(movements)
        
        return {
            'agent_id': agent_id,
            'movements': movements,
            'rooms_visited': rooms_visited,
            'recent_rooms': recent_rooms,
            'tasks_performed': tasks_performed,
            'interactions': interactions,
            'timeline': timeline,
            'time_in_rooms': time_in_rooms,
            'movement_patterns': movement_patterns,
            'total_movements': len(movements),
            'unique_rooms': len(set(rooms_visited))
        }
    
    def _parse_action(self, action: str, timestamp: datetime) -> Optional[Dict]:
        """Parse individual action from trace log"""
        action_lower = action.lower()
        
        # Movement parsing
        movement_match = self.movement_pattern.search(action)
        if movement_match:
            x, y = movement_match.groups()
            room = self._position_to_room(int(x), int(y))
            return {
                'type': 'movement',
                'timestamp': timestamp,
                'position': (int(x), int(y)),
                'room': room,
                'raw_action': action
            }
        
        # Room entry parsing
        room_match = self.room_pattern.search(action_lower)
        if room_match:
            room_name = room_match.group(1).strip()
            return {
                'type': 'room_entry',
                'timestamp': timestamp,
                'room': self._normalize_room_name(room_name),
                'raw_action': action
            }
        
        # Task parsing
        task_match = self.task_pattern.search(action_lower)
        if task_match:
            task_action, task_name, location = task_match.groups()
            return {
                'type': 'task',
                'timestamp': timestamp,
                'task_action': task_action,
                'task_name': task_name.strip(),
                'location': self._normalize_room_name(location.strip()),
                'raw_action': action
            }
        
        # Interaction parsing
        interaction_match = self.interaction_pattern.search(action_lower)
        if interaction_match:
            target = interaction_match.group(1).strip()
            return {
                'type': 'interaction',
                'timestamp': timestamp,
                'target': target,
                'raw_action': action
            }
        
        # Generic action
        return {
            'type': 'other',
            'timestamp': timestamp,
            'action': action_lower,
            'raw_action': action
        }
    
    def _position_to_room(self, x: int, y: int) -> str:
        """Convert grid position to room name (simplified mapping)"""
        # This is a simplified room mapping - you should adjust based on your grid layout
        if x < 5:
            return 'cafeteria'
        elif x < 10:
            return 'weapons' if y < 10 else 'navigation'
        elif x < 15:
            return 'shields' if y < 10 else 'engine'
        else:
            return 'hallway'
    
    def _normalize_room_name(self, room_name: str) -> str:
        """Normalize room name to standard format"""
        room_name_lower = room_name.lower().strip()
        
        for standard_name, variants in self.room_mappings.items():
            if room_name_lower in variants:
                return standard_name
        
        return room_name_lower
    
    def _calculate_room_durations(self, room_timestamps: Dict) -> Dict[str, float]:
        """Calculate time spent in each room in seconds"""
        durations = {}
        
        for room, timestamps in room_timestamps.items():
            if len(timestamps) < 2:
                durations[room] = 0.0
                continue
            
            # Calculate total time as sum of intervals between entries
            total_seconds = 0.0
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                # Cap intervals at reasonable maximum (5 minutes)
                total_seconds += min(interval, 300)
            
            durations[room] = total_seconds
        
        return durations
    
    def _analyze_movement_patterns(self, movements: List[Dict]) -> Dict:
        """Analyze movement patterns for suspicious behavior"""
        if len(movements) < 2:
            return {
                'avg_movement_speed': 0.0,
                'room_transitions': 0,
                'backtracking_score': 0.0,
                'clustering_score': 0.0
            }
        
        # Calculate average movement speed (rooms per minute)
        time_span = (movements[-1]['timestamp'] - movements[0]['timestamp']).total_seconds() / 60
        if time_span > 0:
            avg_speed = len(movements) / time_span
        else:
            avg_speed = 0.0
        
        # Count room transitions
        room_sequence = [m.get('room', 'unknown') for m in movements if 'room' in m]
        transitions = sum(1 for i in range(1, len(room_sequence)) 
                         if room_sequence[i] != room_sequence[i-1])
        
        # Calculate backtracking score (how often agent returns to previous rooms)
        backtracking_score = 0.0
        if len(room_sequence) > 2:
            backtrack_count = 0
            for i in range(2, len(room_sequence)):
                if room_sequence[i] == room_sequence[i-2]:
                    backtrack_count += 1
            backtracking_score = backtrack_count / (len(room_sequence) - 2)
        
        # Calculate clustering score (how concentrated movements are)
        clustering_score = self._calculate_clustering_score(movements)
        
        return {
            'avg_movement_speed': avg_speed,
            'room_transitions': transitions,
            'backtracking_score': backtracking_score,
            'clustering_score': clustering_score,
            'total_time_span_minutes': time_span
        }
    
    def _calculate_clustering_score(self, movements: List[Dict]) -> float:
        """Calculate how clustered the movements are (0 = spread out, 1 = very clustered)"""
        if len(movements) < 3:
            return 0.0
        
        positions = [m['position'] for m in movements if 'position' in m]
        if len(positions) < 3:
            return 0.0
        
        # Calculate variance in positions
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_var = np.var(x_coords) if len(x_coords) > 1 else 0
        y_var = np.var(y_coords) if len(y_coords) > 1 else 0
        
        # Normalize clustering score (lower variance = higher clustering)
        total_var = x_var + y_var
        max_possible_var = 400  # Assuming 20x20 grid
        
        return max(0, 1 - (total_var / max_possible_var))
    
    def verify_location_claim(self, agent_id: int, claimed_room: str, 
                            time_window_minutes: int = 5) -> Dict:
        """Verify if an agent's location claim matches their trace log"""
        try:
            with open(f"agent_{agent_id}_trace.log", "r") as f:
                trace_content = f.read()
        except FileNotFoundError:
            return {
                'verified': False,
                'confidence': 0.0,
                'reason': 'No trace log found',
                'evidence': []
            }
        
        trace_data = self.parse_trace_log(trace_content, agent_id)
        normalized_claim = self._normalize_room_name(claimed_room)
        
        # Check recent rooms
        recent_rooms = trace_data['recent_rooms']
        if normalized_claim in recent_rooms:
            return {
                'verified': True,
                'confidence': 0.9,
                'reason': 'Room found in recent movement history',
                'evidence': [f"Recent rooms: {recent_rooms}"]
            }
        
        # Check time spent in claimed room
        time_in_rooms = trace_data['time_in_rooms']
        if normalized_claim in time_in_rooms and time_in_rooms[normalized_claim] > 30:
            return {
                'verified': True,
                'confidence': 0.7,
                'reason': 'Significant time spent in claimed room',
                'evidence': [f"Time in {normalized_claim}: {time_in_rooms[normalized_claim]:.1f}s"]
            }
        
        # Check all rooms visited
        if normalized_claim in trace_data['rooms_visited']:
            return {
                'verified': True,
                'confidence': 0.5,
                'reason': 'Room visited at some point',
                'evidence': [f"Total rooms visited: {len(trace_data['rooms_visited'])}"]
            }
        
        return {
            'verified': False,
            'confidence': 0.8,
            'reason': 'No evidence of visiting claimed room',
            'evidence': [f"Recent rooms: {recent_rooms}", f"All rooms: {set(trace_data['rooms_visited'])}"]
        }
    
    def detect_suspicious_patterns(self, agent_id: int) -> Dict:
        """Detect suspicious movement patterns that might indicate deceptive behavior"""
        try:
            with open(f"agent_{agent_id}_trace.log", "r") as f:
                trace_content = f.read()
        except FileNotFoundError:
            return {'suspicious_score': 0.0, 'patterns': [], 'confidence': 0.0}
        
        trace_data = self.parse_trace_log(trace_content, agent_id)
        patterns = trace_data['movement_patterns']
        
        suspicious_score = 0.0
        detected_patterns = []
        
        # High backtracking might indicate evasive behavior
        if patterns['backtracking_score'] > 0.4:
            suspicious_score += 0.3
            detected_patterns.append(f"High backtracking: {patterns['backtracking_score']:.2f}")
        
        # Very low movement might indicate hiding
        if patterns['avg_movement_speed'] < 0.5:
            suspicious_score += 0.2
            detected_patterns.append(f"Low activity: {patterns['avg_movement_speed']:.2f} moves/min")
        
        # Very high movement might indicate erratic behavior
        if patterns['avg_movement_speed'] > 10:
            suspicious_score += 0.2
            detected_patterns.append(f"Hyperactive: {patterns['avg_movement_speed']:.2f} moves/min")
        
        # High clustering in one area might indicate camping
        if patterns['clustering_score'] > 0.8:
            suspicious_score += 0.2
            detected_patterns.append(f"High clustering: {patterns['clustering_score']:.2f}")
        
        # Few room transitions might indicate staying in one area
        if patterns['room_transitions'] < 2:
            suspicious_score += 0.1
            detected_patterns.append(f"Low exploration: {patterns['room_transitions']} transitions")
        
        return {
            'suspicious_score': min(suspicious_score, 1.0),
            'patterns': detected_patterns,
            'confidence': 0.7,
            'movement_summary': patterns
        }
    
    def compare_agent_traces(self, agent1_id: int, agent2_id: int) -> Dict:
        """Compare traces of two agents to verify alibi claims"""
        try:
            with open(f"agent_{agent1_id}_trace.log", "r") as f:
                trace1 = f.read()
            with open(f"agent_{agent2_id}_trace.log", "r") as f:
                trace2 = f.read()
        except FileNotFoundError:
            return {
                'alibi_verified': False,
                'confidence': 0.0,
                'reason': 'Missing trace logs',
                'shared_locations': [],
                'temporal_overlap': 0.0
            }
        
        data1 = self.parse_trace_log(trace1, agent1_id)
        data2 = self.parse_trace_log(trace2, agent2_id)
        
        # Find shared rooms
        rooms1 = set(data1['rooms_visited'])
        rooms2 = set(data2['rooms_visited'])
        shared_rooms = rooms1.intersection(rooms2)
        
        # Calculate temporal overlap (simplified)
        timeline1 = data1['timeline']
        timeline2 = data2['timeline']
        
        temporal_overlap = self._calculate_temporal_overlap(timeline1, timeline2, shared_rooms)
        
        # Determine if alibi is plausible
        alibi_verified = len(shared_rooms) > 0 and temporal_overlap > 0.3
        confidence = min(temporal_overlap + 0.2, 1.0) if alibi_verified else temporal_overlap
        
        return {
            'alibi_verified': alibi_verified,
            'confidence': confidence,
            'reason': f"Shared {len(shared_rooms)} rooms with {temporal_overlap:.2f} temporal overlap",
            'shared_locations': list(shared_rooms),
            'temporal_overlap': temporal_overlap
        }
    
    def _calculate_temporal_overlap(self, timeline1: List[Dict], 
                                  timeline2: List[Dict], 
                                  shared_rooms: set) -> float:
        """Calculate temporal overlap between two agent timelines in shared locations"""
        if not shared_rooms or not timeline1 or not timeline2:
            return 0.0
        
        # Get events in shared rooms
        events1 = [e for e in timeline1 if e.get('room') in shared_rooms]
        events2 = [e for e in timeline2 if e.get('room') in shared_rooms]
        
        if not events1 or not events2:
            return 0.0
        
        # Simple overlap calculation based on time windows
        overlap_score = 0.0
        for event1 in events1:
            for event2 in events2:
                if event1.get('room') == event2.get('room'):
                    time_diff = abs((event1['timestamp'] - event2['timestamp']).total_seconds())
                    if time_diff < 300:  # Within 5 minutes
                        overlap_score += max(0, 1 - (time_diff / 300))
        
        return min(overlap_score / max(len(events1), len(events2)), 1.0)
    
    def generate_trace_summary(self, agent_id: int) -> Dict:
        """Generate comprehensive trace summary for an agent"""
        try:
            with open(f"agent_{agent_id}_trace.log", "r") as f:
                trace_content = f.read()
        except FileNotFoundError:
            return {'error': 'No trace log found', 'agent_id': agent_id}
        
        trace_data = self.parse_trace_log(trace_content, agent_id)
        suspicious_patterns = self.detect_suspicious_patterns(agent_id)
        
        return {
            'agent_id': agent_id,
            'summary': {
                'total_movements': trace_data['total_movements'],
                'unique_rooms_visited': trace_data['unique_rooms'],
                'most_visited_rooms': self._get_most_visited_rooms(trace_data),
                'movement_patterns': trace_data['movement_patterns'],
                'suspicious_score': suspicious_patterns['suspicious_score'],
                'detected_patterns': suspicious_patterns['patterns']
            },
            'detailed_data': trace_data
        }
    
    def _get_most_visited_rooms(self, trace_data: Dict) -> List[Tuple[str, int]]:
        """Get most frequently visited rooms"""
        room_counts = {}
        for room in trace_data['rooms_visited']:
            room_counts[room] = room_counts.get(room, 0) + 1
        
        return sorted(room_counts.items(), key=lambda x: x[1], reverse=True)[:5]