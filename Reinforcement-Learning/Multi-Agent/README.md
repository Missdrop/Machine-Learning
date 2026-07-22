# UAV-MEC-MADRL

Multi-agent reinforcement learning for UAV-assisted mobile edge computing, implemented with MADDPG.

This project simulates a wireless offloading scenario with:

- multiple UAV agents
- clustered user devices (UEs)
- fog nodes for edge computing
- centralized training with decentralized execution

## What It Does

- Models a 2D UAV movement environment with fixed altitude.
- Connects UEs to the nearest UAV within communication range.
- Uses MADDPG to train one actor-critic pair per UAV.
- Supports checkpoint saving and loading for all agents.

## Repository Structure

- `environment.py`: Gymnasium environment for UAV, UE, and fog-node simulation.
- `entities.py`: Basic entity classes for UAVs, UEs, and fog nodes.
- `agents.py`: Single-agent actor-critic wrapper.
- `networks.py`: Actor and critic network definitions.
- `memory_replay.py`: Replay buffer for joint multi-agent transitions.
- `maddpg.py`: MADDPG trainer and learning loop.
- `train.ipynb`: Notebook entry point for training and experiments.

## Requirements

- Python 3.10+
- torch
- numpy
- gymnasium
- matplotlib
- pandas
- kagglehub
- torchvision
- livelossplot

## Installation

This repository now manages its own isolated `uv` environment. Run the following from inside this folder:

```bash
uv sync
```

If you want a clean local environment, create and activate a project virtual environment with `uv venv` first, then
install the synced dependencies.

## Training

Open `train.ipynb` and run the cells in order.

The notebook is expected to:

1. Create the environment.
2. Initialize the MADDPG agents.
3. Collect transitions in the replay buffer.
4. Train actors and critics.
5. Save checkpoints for later evaluation.

## Notes

- UAVs move in the x-y plane only, while altitude is fixed.
- UEs may be grouped into clusters to create a more structured traffic pattern.
- The project is suitable for experimenting with multi-agent coordination, offloading policies, and edge-computing
  placement strategies.
