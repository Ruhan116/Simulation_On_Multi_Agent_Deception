# Multi-Agent Deception Simulation

A discrete event simulation framework for studying deception in multi-agent systems, implemented using the Mesa agent-based modeling library. This project simulates an "Among Us"-like environment where agents (crewmates and imposters) interact, communicate, and make decisions based on their observations and reasoning.

## Overview

This simulation models social deception dynamics using various language models (LLMs) as the decision-making engine for agents. The framework allows for:

- Simulating interactions between crewmates and imposters in a spatial environment
- Analyzing deceptive behavior and detection strategies
- Benchmarking different LLM models on their deception capabilities
- Visualizing agent interactions and decision processes

## Features

- **Multi-Agent Environment**: Simulates multiple agents with different roles (crewmates and imposters)
- **Spatial Reasoning**: Agents navigate through rooms and hallways with awareness of their surroundings
- **LLM-Powered Reasoning**: Agents use language models to make decisions and generate arguments
- **Discussion & Voting System**: Simulates group discussions and voting mechanics
- **Benchmarking Tools**: Comprehensive tools to evaluate performance across different LLM models
- **Interactive Visualization**: Web-based visualization of the simulation

## Supported LLM Models

The simulation supports multiple language model providers:
- OpenAI (GPT models)
- Google Gemini
- Groq (Llama models)
- Mistral AI

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment tool (venv, conda, etc.)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Ruhan116/Discrete_Event_Simulation_On_Multi_Agent_Deception.git
   cd Simulation_On_Multi_Agent_Deception
   ```

2. Navigate to the mesa-env directory:
   ```
   cd mesa-env
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up API keys:
   Create a `.env` file in the mesa-env directory with your API keys:
   ```
   OPENAI_KEY=your_openai_key
   GEMINI_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   MISTRAL_API_KEY=your_mistral_key
   ```

### Running the Simulation

To run the interactive visualization:
```
python app.py
```

This will start a local web server. Open your browser and navigate to the URL displayed in the console (typically http://localhost:port).

### Running Benchmarks

To run benchmarks across different LLM models:
```
python benchmark_runner.py
```

You can configure which models to benchmark by editing the `llm_configs` parameter in the `benchmark_runner.py` file.

## Project Structure

- `agents.py`: Defines agent classes (Crewmate, Imposter)
- `model.py`: Main simulation model and logic
- `app.py`: Web visualization server
- `benchmark_runner.py`: Tools for running and analyzing benchmarks
- `llm_handler.py`: Base LLM integration interface
- `gemini_handler.py`, etc.: Specific LLM provider implementations
- `prompts.json`: Standardized prompts for agent reasoning

## Benchmarking

The benchmarking system evaluates:
- Win rates for imposters vs. crewmates
- Statement classification (truth, deception, etc.)
- Decision accuracy
- Agent behavior patterns

Results are saved as JSON files in the `outputs` directory.

## License

[Include license information here]

## Acknowledgements

- [Mesa](https://mesa.readthedocs.io/) - Agent-based modeling framework
- [OpenAI](https://openai.com/), [Google](https://deepmind.google/technologies/gemini/), [Groq](https://groq.com/), and [Mistral AI](https://mistral.ai/) for their LLM APIs
