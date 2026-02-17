# Falling Blocks (REINFORCE)

This project implements a custom Reinforcement Learning environment where an agent must survive as long as possible by avoiding falling blocks in a 1D discrete grid.  
The agent is trained using the **REINFORCE policy gradient algorithm**.

---

## Game Description

We consider a **one-dimensional discrete grid** of fixed width `W`. The agent is located at the bottom row and can move left or right to avoid falling blocks.

At each timestep, a block of some width falls down from the top of the grid. If the falling block reaches the agent level and overlaps the agent position, the episode ends.

The goal of the agent is to **survive as long as possible**.

---

## State Space

The environment state is represented as:

$$
s = (x_{agent}, x_{left}, x_{right}, y_{block})
$$

Where:

- $x_{agent}$ — agent position on the grid
- $x_{left}$ — left coordinate of the falling block
- $x_{right}$ — right coordinate of the falling block
- $y_{block}$ — vertical coordinate of the falling block

---

## Action Space

The agent has 3 discrete actions:

- `0` — move left
- `1` — stay
- `2` — move right

---

## Reward Function

Baseline reward scheme:

- **+1** if the block lands without hitting the agent
- **−10** if the block hits the agent (collision)

The episode terminates when collision happens.

---

## Training Algorithm (REINFORCE)

We train a stochastic policy $\pi_\theta(a|s)$ parameterized by a neural network.

For each episode we collect a trajectory:

$$
(s_0, a_0), (s_1, a_1), ..., (s_\tau, a_\tau)
$$

Discounted returns:

$$
G_t = \sum_{k=t}^{\tau-1} \gamma^{k} r_k
$$

REINFORCE objective:

$$
J(\theta) = \mathbb{E}\left[\sum_{t=0}^{\tau-1} G_t \log \pi_\theta(a_t|s_t)\right]
$$

Loss (minimized in practice):

$$
L(\theta) = -\sum_{t=0}^{\tau-1} G_t \log \pi_\theta(a_t|s_t)
$$

---

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
# Access at http://localhost:8889 (or 8888 if port is available)
```

### Using Makefile

```bash
make build      # Build Docker image
make train      # Run training
make eval       # Run evaluation
make jupyter    # Start Jupyter Lab
make shell      # Interactive shell
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

---

## Baseline Training Results


![Baseline training curve](artifacts/plots/training_baseline.png)

---

## Training Improvements

During experiments we observed that the baseline REINFORCE implementation was unstable: the agent often collapsed to a single action and failed to learn an effective strategy.  
To fix this, we tested multiple modifications.

### 1) Returns Normalization (Standardizing Returns)

Instead of using raw returns \(G_t\), we normalize them inside each episode:

\[
\hat{G}_t = \frac{G_t - \mu}{\sigma + \varepsilon}
\]

Where \(\mu\) is the mean return over the episode and \(\sigma\) is the standard deviation over the episode.  
This improves stability by keeping gradient magnitudes consistent across episodes.

### 2) Entropy Bonus (Encouraging Exploration)

A common failure mode of REINFORCE is **policy collapse**, when the agent becomes overconfident and starts selecting only one action.  
To prevent this, we add entropy regularization:

\[
H(\pi(\cdot|s_t)) = -\sum_a \pi(a|s_t)\log \pi(a|s_t)
\]

Final loss:

\[
L(\theta) = -\sum_{t} \hat{G}_t \log \pi_\theta(a_t|s_t)
\;-\;
\beta \sum_t H(\pi_\theta(\cdot|s_t))
\]

Where \(\beta\) is the entropy coefficient. This significantly improved exploration and prevented the agent from getting stuck.

### 3) Improved State Representation (Relative Coordinates)

The original state used absolute coordinates:

\[
(x_{agent}, x_{left}, x_{right}, y_{block})
\]

Learning from absolute positions was slow. We changed the state to relative distances:

\[
s = (d_{left}, d_{right}, y_{block})
\]

Where:

\[
d_{left} = x_{agent} - x_{left}, \quad
d_{right} = x_{right} - x_{agent}
\]

This makes the policy more position-invariant and improves learning speed.

### 4) Reward Shaping (Better Goal Signal)

Baseline reward:

- +1 for each successfully avoided block
- −10 for collision

Modified reward scheme:

- **+0.1** for each survived timestep
- **+10.0** for each successfully avoided block
- **−10.0** for collision

This provides a clearer training signal and improves learning speed.

---

## Comparison of Training Modifications

> Replace the path below with your actual saved plot.

![Training comparison](artifacts/plots/training_comparison.png)

The comparison plot contains reward vs episode for:

- Baseline REINFORCE
- Returns normalization
- Entropy bonus
- Relative state representation
- Reward shaping
- Combined final version

---

## Conclusion

We demonstrated that REINFORCE can solve the Dodge Blocks environment, but it requires stabilization.  
The best performance was achieved by combining:

- returns normalization
- entropy bonus
- relative state representation
- reward shaping

These modifications made training faster, more stable, and easier to interpret.

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
