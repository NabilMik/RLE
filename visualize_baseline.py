"""
RLE Mini-Challenge: Visualisierung der Baseline-Ergebnisse

Erstellt Plots für den Bericht:
- Reward-Verteilung (Histogramm + Boxplot)
- Episode Length Verteilung
- Reward über Episoden (für Vergleich mit Lernkurven)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


def load_results(results_dir: str = "results"):
    """Lädt die gespeicherten Baseline-Ergebnisse."""
    metrics_path = os.path.join(results_dir, "baseline_random_metrics.json")
    episodes_path = os.path.join(results_dir, "baseline_random_episodes.csv")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episodes_df = pd.read_csv(episodes_path)
    
    return metrics, episodes_df


def plot_baseline_results(metrics: dict, episodes_df: pd.DataFrame, 
                          output_dir: str = "results"):
    """Erstellt alle Visualisierungen für die Baseline."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Baseline Evaluation: Random Agent - Space Invaders', 
                 fontsize=14, fontweight='bold')
    
    rewards = episodes_df['reward'].values
    lengths = episodes_df['length'].values
    episodes = episodes_df['episode'].values
    
    # 1. Reward Histogramm
    ax1 = axes[0, 0]
    ax1.hist(rewards, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(metrics['reward']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['reward']['mean']:.1f}")
    ax1.axvline(metrics['reward']['median'], color='orange', linestyle='--', 
                linewidth=2, label=f"Median: {metrics['reward']['median']:.1f}")
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Boxplot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(rewards, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Reward Boxplot')
    ax2.set_xticklabels(['Random Agent'])
    
    # Statistiken als Text hinzufügen
    stats_text = (f"Mean: {metrics['reward']['mean']:.1f}\n"
                  f"Std: {metrics['reward']['std']:.1f}\n"
                  f"Min: {metrics['reward']['min']:.0f}\n"
                  f"Max: {metrics['reward']['max']:.0f}")
    ax2.text(1.3, np.median(rewards), stats_text, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward über Zeit (Rolling Average)
    ax3 = axes[1, 0]
    window = 10
    rolling_mean = pd.Series(rewards).rolling(window=window).mean()
    ax3.plot(episodes, rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    ax3.plot(episodes, rolling_mean, color='red', linewidth=2, 
             label=f'Rolling Mean (window={window})')
    ax3.axhline(metrics['reward']['mean'], color='green', linestyle='--', 
                alpha=0.7, label=f"Overall Mean: {metrics['reward']['mean']:.1f}")
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward over Episodes')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Episode Length Distribution
    ax4 = axes[1, 1]
    ax4.hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='forestgreen')
    ax4.axvline(metrics['episode_length']['mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['episode_length']['mean']:.1f}")
    ax4.set_xlabel('Episode Length (Steps)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Length Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    plot_path = os.path.join(output_dir, "baseline_random_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots gespeichert: {plot_path}")
    
    plt.show()
    
    return fig


def create_comparison_table(metrics: dict):
    """
    Erstellt eine Vergleichstabelle für den Bericht.
    Diese kann später mit RL-Agenten erweitert werden.
    """
    table_data = {
        'Metric': [
            'Mean Reward',
            'Std Reward', 
            'Min Reward',
            'Max Reward',
            'Median Reward',
            'Mean Episode Length'
        ],
        'Random Agent': [
            f"{metrics['reward']['mean']:.2f}",
            f"{metrics['reward']['std']:.2f}",
            f"{metrics['reward']['min']:.0f}",
            f"{metrics['reward']['max']:.0f}",
            f"{metrics['reward']['median']:.0f}",
            f"{metrics['episode_length']['mean']:.2f}"
        ]
    }
    
    df = pd.DataFrame(table_data)
    print("\n" + "="*50)
    print("VERGLEICHSTABELLE (für Bericht)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    return df


if __name__ == "__main__":
    # Ergebnisse laden
    try:
        metrics, episodes_df = load_results()
        
        # Plots erstellen
        plot_baseline_results(metrics, episodes_df)
        
        # Vergleichstabelle erstellen
        comparison_df = create_comparison_table(metrics)
        comparison_df.to_csv("results/comparison_table.csv", index=False)
        
    except FileNotFoundError:
        print("Keine Baseline-Ergebnisse gefunden!")
        print("Bitte zuerst 'baseline_random_agent.py' ausführen.")
