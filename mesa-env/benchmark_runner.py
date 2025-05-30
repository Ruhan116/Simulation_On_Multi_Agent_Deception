import json
import time
from model import AmongUsModel
from benchmark_manager import BenchmarkManager
from datetime import datetime

class CompleteBenchmarkRunner:
    def __init__(self):
        self.benchmark_manager = BenchmarkManager()
        self.results = []
    
    def run_comprehensive_benchmark(self, 
                                   num_games_per_llm=50, 
                                   llm_types=['gemini', 'openai', 'groq']):
        """Run comprehensive benchmark across multiple LLMs"""
        
        total_games = len(llm_types) * num_games_per_llm
        game_count = 0
        
        for llm_type in llm_types:
            print(f"\n=== Benchmarking {llm_type} ({num_games_per_llm} games) ===")
            
            for game_num in range(num_games_per_llm):
                game_count += 1
                print(f"Game {game_count}/{total_games} - {llm_type} #{game_num + 1}")
                
                try:
                    # Create model with current LLM
                    model = AmongUsModel(
                        num_agents=4, 
                        num_imposters=1, 
                        llm_type=llm_type
                    )
                    
                    # Run game
                    self._run_single_game(model)
                    
                    # Clean up
                    self._cleanup_game_files(model)
                    
                    # Progress update every 10 games
                    if game_count % 10 == 0:
                        self._print_progress_report(game_count, total_games)
                        
                except Exception as e:
                    print(f"Error in game {game_count}: {str(e)}")
                    continue
        
        # Generate final report
        final_report = self.generate_final_report()
        return final_report
    
    def _run_single_game(self, model):
        """Run a single game to completion"""
        max_steps = 1000  # Prevent infinite games
        step_count = 0
        
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
        print(f"Avg Lying Frequency: {current_stats.get('lying_frequency', {}).get('overall', 0):.3f}")
        print(f"Avg Suspicion Accuracy: {current_stats.get('suspicion_accuracy', {}).get('overall', 0):.3f}")
    
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
        
        # Elo ratings
        deception_elo = report.get('deception_elo', {})
        detection_elo = report.get('detection_elo', {})
        print(f"\nELO RATINGS:")
        print(f"  Avg Deception Elo: {deception_elo.get('mean', 1500):.0f}")
        print(f"  Avg Detection Elo: {detection_elo.get('mean', 1500):.0f}")
        
        # Deception metrics
        lying_freq = report.get('lying_frequency', {})
        truth_rate = report.get('truth_telling_rate', {})
        print(f"\nDECEPTION METRICS:")
        print(f"  Lying Frequency: {lying_freq.get('overall', 0):.1%}")
        print(f"  Truth-telling Rate: {truth_rate.get('overall', 1):.1%}")
        
        # Accuracy
        accuracy = report.get('suspicion_accuracy', {})
        print(f"\nACCURACY METRICS:")
        print(f"  Suspicion Accuracy: {accuracy.get('overall', 0):.1%}")
        
        # Speech classification
        speech_class = report.get('speech_classification', {})
        overall_dist = speech_class.get('overall_distribution', {})
        print(f"\nSPEECH CLASSIFICATION:")
        for speech_type, freq in overall_dist.items():
            print(f"  {speech_type}: {freq:.1%}")
        
        print("="*60)

# Usage example:
if __name__ == "__main__":
    runner = CompleteBenchmarkRunner()
    results = runner.run_comprehensive_benchmark(
        num_games_per_llm=25,  # Start with smaller number for testing
        llm_types=['gemini']   # Start with one LLM for testing
    )