"""
RLE Mini-Challenge: Visualisierung aller Experiment-Ergebnisse
==============================================================
Erstellt Vergleichs-Plots für alle DQN-Varianten.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Stil für wissenschaftliche Plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 8)

# Ergebnisse (aus den Trainings)
results = {
    'Random Baseline': {
        'mean': 139.70,
        'std': 86.49,
        'min': 15,
        'max': 530,
        'color': '#808080'
    },
    'DQN Initial': {
        'mean': 486.65,
        'std': 185.77,
        'min': 85,
        'max': 1050,
        'color': '#2ecc71'
    },
    'Double DQN': {
        'mean': 484.80,
        'std': 175.12,
        'min': 75,
        'max': 1025,
        'color': '#3498db'
    },
    'Dueling DQN': {
        'mean': 448.40,
        'std': 179.73,
        'min': 50,
        'max': 890,
        'color': '#9b59b6'
    },
    'Hyperparameter\nTuning': {
        'mean': 337.70,
        'std': 150.0,  # geschätzt
        'min': 50,
        'max': 870,
        'color': '#e74c3c'
    },
    'Larger\nNetwork': {
        'mean': 429.15,
        'std': 160.0,  # geschätzt
        'min': 50,
        'max': 870,
        'color': '#f39c12'
    }
}

def load_json_results(results_dir='results'):
    """Versucht JSON-Dateien zu laden falls vorhanden."""
    json_files = {
        'baseline_random_metrics.json': 'Random Baseline',
        'dqn_initial_results.json': 'DQN Initial',
        'double_dqn_results.json': 'Double DQN',
        'dueling_dqn_results.json': 'Dueling DQN',
        'hyperparameter_tuning_results.json': 'Hyperparameter\nTuning',
        'larger_network_results.json': 'Larger\nNetwork'
    }
    
    results_path = Path(results_dir)
    if results_path.exists():
        for filename, name in json_files.items():
            filepath = results_path / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if 'final_evaluation' in data:
                        eval_data = data['final_evaluation']
                        results[name]['mean'] = eval_data.get('mean', results[name]['mean'])
                        results[name]['std'] = eval_data.get('std', results[name]['std'])
                        results[name]['min'] = eval_data.get('min', results[name]['min'])
                        results[name]['max'] = eval_data.get('max', results[name]['max'])
                    print(f"✓ Geladen: {filename}")
                except Exception as e:
                    print(f"✗ Fehler bei {filename}: {e}")

def plot_mean_comparison():
    """Balkendiagramm: Mean Reward Vergleich."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]
    colors = [results[n]['color'] for n in names]
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Werte über Balken
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Baseline-Linie
    ax.axhline(y=results['Random Baseline']['mean'], color='gray', 
               linestyle='--', linewidth=2, label='Random Baseline')
    
    # Best Result hervorheben
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    ax.set_ylabel('Mean Reward (100 Episoden)')
    ax.set_title('Space Invaders: Vergleich aller DQN-Varianten\n(1M Trainingsschritte)')
    ax.set_ylim(0, max(means) + max(stds) + 100)
    
    plt.tight_layout()
    plt.savefig('results/comparison_mean_reward.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/comparison_mean_reward.pdf', bbox_inches='tight')
    print("✓ Gespeichert: comparison_mean_reward.png/pdf")
    plt.close()

def plot_improvement_over_baseline():
    """Balkendiagramm: Prozentuale Verbesserung über Baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = results['Random Baseline']['mean']
    
    # Ohne Baseline selbst
    names = [n for n in results.keys() if n != 'Random Baseline']
    improvements = [(results[n]['mean'] - baseline) / baseline * 100 for n in names]
    colors = [results[n]['color'] for n in names]
    
    bars = ax.bar(names, improvements, color=colors, edgecolor='black', 
                  linewidth=1.5, alpha=0.85)
    
    # Werte über Balken
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'+{imp:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Verbesserung über Baseline (%)')
    ax.set_title('Prozentuale Verbesserung gegenüber Random Agent')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('results/improvement_over_baseline.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/improvement_over_baseline.pdf', bbox_inches='tight')
    print("✓ Gespeichert: improvement_over_baseline.png/pdf")
    plt.close()

def plot_boxplot_style():
    """Min/Max/Quartile Visualisierung."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = list(results.keys())
    positions = range(len(names))
    
    for i, name in enumerate(names):
        r = results[name]
        color = r['color']
        
        # Min-Max Linie
        ax.vlines(i, r['min'], r['max'], color=color, linewidth=2, alpha=0.5)
        
        # Mean Punkt
        ax.scatter(i, r['mean'], color=color, s=200, zorder=5, edgecolor='black', linewidth=2)
        
        # Std Bereich
        ax.fill_between([i-0.2, i+0.2], 
                        [r['mean']-r['std'], r['mean']-r['std']], 
                        [r['mean']+r['std'], r['mean']+r['std']], 
                        color=color, alpha=0.3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel('Reward')
    ax.set_title('Reward-Verteilung: Mean ± Std mit Min/Max Range')
    
    # Legende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Mean'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.5, label='Min-Max Range'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/reward_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/reward_distribution.pdf', bbox_inches='tight')
    print("✓ Gespeichert: reward_distribution.png/pdf")
    plt.close()

def plot_ranking_horizontal():
    """Horizontales Ranking-Diagramm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sortiert nach Mean
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    names = [r[0] for r in sorted_results]
    means = [r[1]['mean'] for r in sorted_results]
    colors = [r[1]['color'] for r in sorted_results]
    
    bars = ax.barh(names, means, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Werte neben Balken
    for bar, mean in zip(bars, means):
        ax.text(mean + 10, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Ranking-Nummern
    for i, (bar, name) in enumerate(zip(bars, names)):
        ax.text(20, bar.get_y() + bar.get_height()/2,
                f'#{i+1}', ha='left', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.set_xlabel('Mean Reward')
    ax.set_title('Ranking: DQN-Varianten nach Performance')
    ax.set_xlim(0, max(means) + 80)
    
    plt.tight_layout()
    plt.savefig('results/ranking.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/ranking.pdf', bbox_inches='tight')
    print("✓ Gespeichert: ranking.png/pdf")
    plt.close()

def plot_summary_table():
    """Erstellt eine Zusammenfassungs-Tabelle als Bild."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    # Daten für Tabelle
    columns = ['Agent', 'Mean', 'Std', 'Min', 'Max', 'vs. Baseline']
    baseline = results['Random Baseline']['mean']
    
    table_data = []
    for name, r in results.items():
        improvement = ((r['mean'] - baseline) / baseline * 100) if name != 'Random Baseline' else 0
        imp_str = f"+{improvement:.0f}%" if improvement > 0 else "-"
        table_data.append([
            name.replace('\n', ' '),
            f"{r['mean']:.1f}",
            f"{r['std']:.1f}",
            f"{r['min']}",
            f"{r['max']}",
            imp_str
        ])
    
    # Sortiert nach Mean
    table_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    table = ax.table(cellText=table_data, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#f0f0f0']*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header fett
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    # Beste Zeile hervorheben
    for i in range(len(columns)):
        table[(1, i)].set_facecolor('#d4edda')
    
    ax.set_title('Zusammenfassung aller Experimente', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/summary_table.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/summary_table.pdf', bbox_inches='tight')
    print("✓ Gespeichert: summary_table.png/pdf")
    plt.close()

def main():
    """Hauptfunktion: Erstellt alle Visualisierungen."""
    print("=" * 60)
    print("RLE Mini-Challenge: Visualisierung")
    print("=" * 60)
    
    # Versuche JSON-Dateien zu laden
    print("\n Lade Ergebnisse aus JSON-Dateien...")
    load_json_results()
    
    # Erstelle results Ordner falls nicht vorhanden
    Path('results').mkdir(exist_ok=True)
    
    print("\n Erstelle Visualisierungen...")
    
    # Alle Plots erstellen
    plot_mean_comparison()
    plot_improvement_over_baseline()
    plot_boxplot_style()
    plot_ranking_horizontal()
    plot_summary_table()
    
    print("\n" + "=" * 60)
    print(" Alle Visualisierungen erstellt!")
    print("=" * 60)
    print("\nDateien in results/:")
    print("  - comparison_mean_reward.png/pdf")
    print("  - improvement_over_baseline.png/pdf")
    print("  - reward_distribution.png/pdf")
    print("  - ranking.png/pdf")
    print("  - summary_table.png/pdf")

if __name__ == "__main__":
    main()
