# RL Dodge Blocks Project

A Reinforcement Learning project for training an agent to play the Dodge Blocks game using the REINFORCE algorithm.

## Overview

This project implements a REINFORCE-based agent that learns to dodge falling blocks in a grid-based environment. The implementation includes ablation studies, comprehensive logging, and evaluation tools.

## Features

- **REINFORCE Algorithm**: Policy gradient method for training the agent
- **Configurable Environment**: Support for different state representations and reward functions
- **Ablation Studies**: Systematic experiments with different configurations
- **Comprehensive Logging**: Training statistics and checkpoint management
- **Evaluation Tools**: Model evaluation with rendering capabilities
- **Docker Support**: Containerized environment for easy deployment

## Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy
- Pygame (for rendering)
- Pandas, Matplotlib (for analysis)
- Jupyter (for notebooks)

## Installation

### Using Docker (Recommended)

```bash
# Build the Docker image
docker compose build

# Or using Makefile
make build
```

### Local Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python run/train.py --name baseline --episodes 800
```

### Evaluation

```bash
python run/evaluate.py --checkpoint artifacts/checkpoints/best.pt --num_episodes 100 --render
```

### Ablation Study

```bash
bash run_ablation.sh
```

## Project Structure

```
.
├── src/
│   ├── agent/           # REINFORCE agent implementation
│   ├── environment/      # Game environment and renderer
│   ├── training/        # Training loop and logger
│   └── utils/           # Configuration and utilities
├── run/                 # Training and evaluation scripts
├── artifacts/           # Checkpoints, logs, and statistics
├── analysis/            # Jupyter notebooks for analysis
└── tests/               # Unit tests
```

## Configuration

### Training Parameters

- `--name`: Experiment name
- `--norm`: Enable return normalization
- `--entropy`: Entropy coefficient for exploration
- `--state`: State representation mode (`absolute` or `relative`)
- `--reward`: Reward function mode (`basic` or `enhanced`)
- `--episodes`: Number of training episodes
- `--seed`: Random seed for reproducibility

### Environment Configuration

- Grid size: 6×12
- Block width: 1-2 cells
- Fall speed: 1 cell per step
- Action space: 3 actions (left, stay, right)

## Docker Usage

### Build and Run

```bash
# Build image
docker compose build

# Run training
docker compose run --rm train

# Run evaluation
docker compose run --rm evaluate

# Start Jupyter Lab
docker compose up jupyter
# Access at http://localhost:8888
```

### Using Makefile

```bash
make build      # Build Docker image
make train      # Run training
make eval       # Run evaluation
make jupyter    # Start Jupyter Lab
make shell      # Interactive shell
```

## Results

Training results are saved in `artifacts/`:
- `checkpoints/`: Model checkpoints (best.pt, last.pt)
- `logs/`: Training logs and statistics
- `stats/`: Performance metrics
- `ablation/`: Ablation study results

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Agent**: `src/agent/reinforce_agent.py` - REINFORCE algorithm implementation
- **Environment**: `src/environment/game_env.py` - Game logic and state management
- **Training**: `src/training/trainer.py` - Training loop and optimization
- **Config**: `src/utils/config.py` - Configuration dataclasses

## License

This is an educational project.
