"""
RLE Mini-Challenge: Baseline - Random Agent für Space Invaders

Dieser Agent wählt zufällige Aktionen und dient als Baseline für den Vergleich
mit trainierten RL-Agenten.

Evaluierte Metriken:
- Episode Reward (Mean, Std, Min, Max, Median)
- Episode Length (Mean, Std, Min, Max)
- Actions Distribution
- Score over Time (für Lernkurven-Vergleich)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
import os


class RandomAgent:
    """Ein Agent der zufällige Aktionen wählt."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_counts = defaultdict(int)
    
    def select_action(self, observation):
        """Wählt eine zufällige Aktion."""
        action = self.action_space.sample()
        self.action_counts[action] += 1
        return action
    
    def get_action_distribution(self):
        """Gibt die Verteilung der gewählten Aktionen zurück."""
        total = sum(self.action_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(self.action_counts.items())}


def evaluate_baseline(n_episodes: int = 100, render: bool = False, seed: int = 42):
    """
    Evaluiert den Random Agent über n_episodes.
    
    Args:
        n_episodes: Anzahl der Episoden für die Evaluation
        render: Ob das Spiel gerendert werden soll
        seed: Random seed für Reproduzierbarkeit
    
    Returns:
        Dictionary mit allen Evaluationsmetriken
    """
    # Environment Setup
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        render_mode="human" if render else None,
        frameskip=4,  # Standard Frame Skip
        repeat_action_probability=0.0  # Deterministische Aktionen
    )
    
    # Seed setzen für Reproduzierbarkeit
    np.random.seed(seed)
    
    # Agent initialisieren
    agent = RandomAgent(env.action_space)
    
    # Metriken sammeln
    episode_rewards = []
    episode_lengths = []
    episode_data = []  # Für detaillierte Analyse
    
    print(f"{'='*60}")
    print(f"Baseline Evaluation: Random Agent")
    print(f"{'='*60}")
    print(f"Environment: ALE/SpaceInvaders-v5")
    print(f"Action Space: {env.action_space} ({env.action_space.n} actions)")
    print(f"Observation Space: {env.observation_space}")
    print(f"Episodes: {n_episodes}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    for episode in range(n_episodes):
        observation, info = env.reset(seed=seed + episode)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_data.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length
        })
        
        # Fortschritt anzeigen
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.0f} | "
                  f"Length: {episode_length} | "
                  f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    
    env.close()
    
    # Metriken berechnen
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    metrics = {
        'agent_type': 'Random Agent',
        'n_episodes': n_episodes,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        
        # Reward Statistiken
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
            'q25': float(np.percentile(rewards, 25)),
            'q75': float(np.percentile(rewards, 75)),
        },
        
        # Episode Length Statistiken
        'episode_length': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths)),
            'median': float(np.median(lengths)),
        },
        
        # Action Distribution
        'action_distribution': agent.get_action_distribution(),
        
        # Rohdaten für weitere Analyse
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    return metrics, episode_data


def print_results(metrics: dict):
    """Gibt die Ergebnisse formatiert aus."""
    print(f"\n{'='*60}")
    print("BASELINE ERGEBNISSE: Random Agent")
    print(f"{'='*60}\n")
    
    print("REWARD STATISTIKEN:")
    print(f"  Mean:     {metrics['reward']['mean']:.2f}")
    print(f"  Std:      {metrics['reward']['std']:.2f}")
    print(f"  Min:      {metrics['reward']['min']:.0f}")
    print(f"  Max:      {metrics['reward']['max']:.0f}")
    print(f"  Median:   {metrics['reward']['median']:.0f}")
    print(f"  Q25:      {metrics['reward']['q25']:.0f}")
    print(f"  Q75:      {metrics['reward']['q75']:.0f}")
    
    print("\nEPISODE LENGTH STATISTIKEN:")
    print(f"  Mean:     {metrics['episode_length']['mean']:.2f}")
    print(f"  Std:      {metrics['episode_length']['std']:.2f}")
    print(f"  Min:      {metrics['episode_length']['min']:.0f}")
    print(f"  Max:      {metrics['episode_length']['max']:.0f}")
    print(f"  Median:   {metrics['episode_length']['median']:.0f}")
    
    print("\nACTION DISTRIBUTION:")
    action_names = {
        0: "NOOP",
        1: "FIRE", 
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }
    for action, prob in metrics['action_distribution'].items():
        name = action_names.get(action, f"Action {action}")
        print(f"  {name}: {prob*100:.1f}%")
    
    print(f"\n{'='*60}")


def save_results(metrics: dict, episode_data: list, output_dir: str = "results"):
    """Speichert die Ergebnisse in Dateien."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Metriken als JSON (ohne Rohdaten für bessere Lesbarkeit)
    metrics_clean = {k: v for k, v in metrics.items() 
                     if k not in ['episode_rewards', 'episode_lengths']}
    
    # Action Distribution Keys zu strings konvertieren (für JSON)
    if 'action_distribution' in metrics_clean:
        metrics_clean['action_distribution'] = {
            str(k): v for k, v in metrics_clean['action_distribution'].items()
        }
    
    json_path = os.path.join(output_dir, "baseline_random_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"Metriken gespeichert: {json_path}")
    
    # Episode-Daten als CSV
    df = pd.DataFrame(episode_data)
    csv_path = os.path.join(output_dir, "baseline_random_episodes.csv")
    df.to_csv(csv_path, index=False)
    print(f"Episode-Daten gespeichert: {csv_path}")
    
    # Alle Rewards als NumPy für spätere Analyse
    np_path = os.path.join(output_dir, "baseline_random_rewards.npy")
    np.save(np_path, np.array(metrics['episode_rewards']))
    print(f"Rewards-Array gespeichert: {np_path}")


if __name__ == "__main__":
    # Evaluation durchführen
    metrics, episode_data = evaluate_baseline(
        n_episodes=100,  # Mindestens 100 wie in der Aufgabe gefordert
        render=False,    # Auf True setzen um das Spiel zu sehen
        seed=42
    )
    
    # Ergebnisse ausgeben
    print_results(metrics)
    
    # Ergebnisse speichern
    save_results(metrics, episode_data)
    
    print("\nBaseline-Evaluation abgeschlossen!")
    print("Diese Ergebnisse dienen als Referenz für den Vergleich mit RL-Agenten.")
