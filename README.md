# RL Car Race — Project PG-4

Reinforcement Learning project for the AA 2025-26 course at Università degli Studi di Milano.

## Goal

Train an autonomous agent to drive a Formula 1 car around a circuit as fast as possible without going off track, using policy gradient methods with deep neural network policies.

## Approach

- **Environment**: custom MDP with continuous state (position relative to circuit, velocity) and continuous actions (acceleration, steering)
- **Circuit**: TBD
- **Policy**: parametric Gaussian/deterministic policy implemented as a MLP, with configurable depth and width
- **Algorithm**: TBD (policy gradient-based)

## Experiments

The main comparison is across policy networks of different sizes, measuring:
- Final performance
- Episodes to convergence
- Wall-clock time to convergence

## Structure

```
.
├── env/          # custom environment
├── policy/       # neural network policy definitions
├── train.py      # training script
├── evaluate.py   # evaluation script
└── results/      # saved runs and plots
```

## Usage

_To be filled in._

## Course Info

- **Course**: Reinforcement Learning, AA 2025-26
- **Project ID**: PG-4
- **Instructors**: Alfio Ferrara, Matteo Papini, Nicolò Cesa-Bianchi