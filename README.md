# RLE Mini-Challenge: DQN für Space Invaders

Deep Reinforcement Learning Agenten für das Atari-Spiel Space Invaders.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Baseline
python baseline_random_agent.py

# DQN Initial
python dqn_initial.py

# Erweiterungen
python double_dqn.py
python dueling_dqn.py
python hyperparameter_tuning.py
python larger_network.py
```

## Ergebnisse anschauen

```bash
# TensorBoard für Lernkurven
tensorboard --logdir runs

# Visualisierungen
python visualize_results.py
python visualize_baseline.py
```

## Projektstruktur

```
rle-mini-challenge/
├── baseline_random_agent.py   # Random Agent Baseline
├── dqn_initial.py             # Initialer DQN-Ansatz
├── double_dqn.py              # Erweiterung 1: Double DQN
├── dueling_dqn.py             # Erweiterung 2: Dueling DQN
├── hyperparameter_tuning.py   # Erweiterung 3: HP-Tuning
├── larger_network.py          # Erweiterung 4: Grösseres Netzwerk
├── results/                   # Evaluationsergebnisse (JSON)
├── runs/                      # TensorBoard Logs
├── videos/                    # Aufgezeichnete Gameplay-Videos
├── bericht/                    # Bericht
└── requirements.txt           # Python Dependencies
```

## Ergebnisse

| Agent | Mean Reward | vs. Baseline |
|-------|-------------|--------------|
| DQN Initial | 486.6 | +248% |
| Double DQN | 484.8 | +247% |
| Dueling DQN | 448.4 | +221% |
| Larger Network | 429.1 | +207% |
| Hyperparameter-Tuning | 337.7 | +142% |
| Random Baseline | 139.7 | - |

## Hardware

- Apple MacBook M2 (CPU)
- Trainingszeit: ca. 3-4 Stunden pro Agent
