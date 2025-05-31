import json
import time
from model import AmongUsModel
from benchmark_manager import BenchmarkManager
from datetime import datetime
import uuid
from agents import Imposter


class CompleteBenchmarkRunner:
    def __init__(self):
        self.benchmark_manager = BenchmarkManager()
        self.results = []
    
    def run_comprehensive_benchmark(self, 
                                   num_games_per_llm=50, 
                                   llm_configs=None):
        """
        Run comprehensive benchmark across multiple LLMs.
        llm_configs: list of dicts, each with keys 'type' and 'model', e.g.
            [{'type': 'gemini', 'model': 'gemini-2.0-flash'}, ...]
        """
        if llm_configs is None:
            llm_configs = [
                {'type': 'gemini', 'model': 'gemini-2.0-flash'},
                {'type': 'openai', 'model': 'gpt-3.5-turbo'},
                {'type': 'groq', 'model': 'llama3-70b-8192'},
                {'type': 'mistral', 'model': 'mistral-large-latest'}
            ]

        results = []
        for llm_cfg in llm_configs:
            # Clear the database before each LLM config
            self.benchmark_manager.clear_database()

            llm_type = llm_cfg['type']
            llm_model = llm_cfg['model']
            print(f"\n=== Benchmarking {llm_type} ({llm_model}) ({num_games_per_llm} games) ===")

            total_games = num_games_per_llm
            game_count = 0

            for game_num in range(num_games_per_llm):
                game_count += 1
                print(f"Game {game_count}/{total_games} - {llm_type} ({llm_model}) #{game_num + 1}")
                try:
                    model = AmongUsModel(
                        num_agents=4, 
                        num_imposters=1, 
                        llm_type=llm_type,
                        llm_model=llm_model
                    )
                    self._run_single_game(model)
                    self._cleanup_game_files(model)
                    if game_count % 10 == 0:
                        self._print_progress_report(game_count, total_games)
                except Exception as e:
                    print(f"Error in game {game_count}: {str(e)}")
                    continue

            # Generate and print/save report for this LLM
            final_report = self.generate_final_report()
            results.append(final_report)

        return results
    
    def _run_single_game(self, model):
        """Run a single game to completion"""
        max_steps = 1000
        step_count = 0

        # Initialize benchmark manager in model
        model.benchmark_manager = self.benchmark_manager

        model.game_id = str(uuid.uuid4())

        while model.running and step_count < max_steps:
            model.step()
            step_count += 1

            # Add delay for API rate limiting
            if step_count % 5 == 0:
                time.sleep(0.2)

        if model.running:
            print(f"Game timed out after {max_steps} steps")
            # Force game completion for benchmarking
            model.winner = "Timeout"
            model.benchmark_manager.record_game_completion(model, model.game_id)
        else:
            print(f"Game completed: {model.winner} won in {step_count} steps!")
    
    def _cleanup_game_files(self, model):
        """Clean up trace files after each game"""
        import os
        from agents import Crewmate
        
        for agent in model.schedule.agents:
            if isinstance(agent, Crewmate):
                try:
                    # Close trace file if open
                    if hasattr(agent, '_trace_file') and agent._trace_file:
                        agent._trace_file.close()
                    
                    # Remove trace file
                    trace_file_path = f"agent_{agent.unique_id}_trace.log"
                    if os.path.exists(trace_file_path):
                        os.remove(trace_file_path)
                except Exception as e:
                    print(f"Cleanup error for agent {agent.unique_id}: {e}")
    
    def _print_progress_report(self, completed: int, total: int):
        """Print progress report with current metrics"""
        progress = (completed / total) * 100
        print(f"\n--- Progress: {completed}/{total} ({progress:.1f}%) ---")
        
        # Get current benchmark stats
        current_stats = self.benchmark_manager.calculate_all_benchmarks()
        print(f"Current Win Rates: {current_stats.get('win_rates', {})}")
        
        # Print speech distribution properly
        speech_dist = current_stats.get('speech_classification', {}).get('overall_distribution', {})
        if speech_dist:
            print("Current Speech Distribution:")
            for speech_type, freq in speech_dist.items():
                print(f"  {speech_type}: {freq:.1%}")
    
    def generate_final_report(self):
        """Generate comprehensive final benchmark report"""
        print("\n=== GENERATING FINAL BENCHMARK REPORT ===")
        
        # Calculate all benchmarks
        report = self.benchmark_manager.calculate_all_benchmarks()
        
        # Add metadata
        report['metadata'] = {
            'generation_time': datetime.now().isoformat(),
            'total_games': self._count_total_games(),
            'llm_types_tested': self._get_tested_llm_types()
        }
        
        # Save to file
        filename = f"comprehensive_benchmark_report_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_final_summary(report)
        
        return report
    
    def _count_total_games(self):
        """Count total games in database"""
        import sqlite3
        conn = sqlite3.connect(self.benchmark_manager.db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def _get_tested_llm_types(self):
        """Get list of tested LLM types"""
        import sqlite3
        conn = sqlite3.connect(self.benchmark_manager.db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT llm_type FROM games")
        types = [row[0] for row in cursor.fetchall()]
        conn.close()
        return types
    
    def _print_final_summary(self, report):
        """Print final benchmark summary"""
        print("\n" + "="*60)
        print("FINAL BENCHMARK RESULTS")
        print("="*60)
        
        # Win rates
        win_rates = report.get('win_rates', {})
        print(f"\nWIN RATES:")
        print(f"  Imposter: {win_rates.get('imposter', 0):.1%}")
        print(f"  Crewmate: {win_rates.get('crewmate', 0):.1%}")
        print(f"  Total Games: {win_rates.get('total_games', 0)}")
        
        # Speech classification
        speech_class = report.get('speech_classification', {})
        overall_dist = speech_class.get('overall_distribution', {})
        print(f"\nSPEECH DISTRIBUTION:")
        for speech_type, freq in overall_dist.items():
            print(f"  {speech_type}: {freq:.1%}")
        
        print("="*60)
    
    def _analyze_game_statements(self, model, game_id: str):
        """Analyze all statements from a completed game"""
        # Ensure model.trace_logs exists and is a list of step logs
        for step_log in getattr(model, "trace_logs", []):
            for entry in step_log.get("statements", []):
                agent = model.get_agent_by_id(entry['agent_id'])
                self.record_statement(
                    agent=agent,
                    argument=entry,
                    game_id=game_id,
                    step=entry.get("step", 0),
                    room=entry.get("room", ""),
                    game_context={
                        'actual_imposters': [a.unique_id for a in model.schedule.agents if isinstance(a, Imposter)],
                        'innocent_players': [a.unique_id for a in model.schedule.agents if not isinstance(a, Imposter)]
                    }
                )

# Usage example:
if __name__ == "__main__":
    runner = CompleteBenchmarkRunner()
    results = runner.run_comprehensive_benchmark(
        num_games_per_llm=1,
        llm_configs=[
            # {'type': 'mistral', 'model': 'mistral-large-latest'},
            # {'type': 'gemini', 'model': 'gemini-2.0-flash'},
            {'type': 'groq', 'model': 'llama3-70b-8192'}
        ]
    )