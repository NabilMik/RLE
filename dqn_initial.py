"""
RLE Mini-Challenge: Initialer DQN-Ansatz für Space Invaders

Basierend auf CleanRL DQN Implementation.
Angepasst für konsistente Evaluation und Vergleichbarkeit.

BEGRÜNDUNG DER ALGORITHMUS-WAHL:
================================
DQN (Deep Q-Network) wurde gewählt, weil:
1. Value-Based Methode - gut geeignet für diskrete Aktionsräume (6 Aktionen in Space Invaders)
2. Bewährter Algorithmus für Atari-Spiele (DeepMind 2015)
3. Viele Erweiterungsmöglichkeiten: Double DQN, Dueling DQN, Prioritized Experience Replay
4. Gut dokumentiert und verstanden in der Literatur
5. Stabiles Training durch Experience Replay und Target Network

HYPERPARAMETER:
===============
Siehe Args Dataclass unten für vollständige Dokumentation.
"""

import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # Experiment Identifikation
    exp_name: str = "dqn_initial"
    """Name dieses Experiments - wird für Vergleiche verwendet"""
    seed: int = 42
    """Random Seed für Reproduzierbarkeit"""
    torch_deterministic: bool = True
    """Deterministische PyTorch Operationen"""
    cuda: bool = True
    """CUDA verwenden falls verfügbar"""
    
    # Logging & Speichern
    capture_video: bool = True
    """Videos der Agent-Performance aufnehmen"""
    save_model: bool = True
    """Modell speichern nach Training"""
    
    # Environment
    env_id: str = "ALE/SpaceInvaders-v5"
    """Gym Environment ID"""
    
    # === TRAININGSBUDGET ===
    total_timesteps: int = 1_000_000
    """
    Trainingsbudget: 1 Million Timesteps
    - Mit Frame Skip=4 entspricht dies 4 Millionen Spielframes
    - Agent zeigt bereits deutliche Konvergenz (3x besser als Baseline)
    - Alle Experimente (Initial + Erweiterungen) nutzen dasselbe Budget
    """
    
    # === DQN HYPERPARAMETER ===
    learning_rate: float = 1e-4
    """Lernrate für Adam Optimizer"""
    
    buffer_size: int = 100_000
    """
    Replay Buffer Größe
    - Reduziert von 1M auf 100K für Memory-Effizienz
    - Ausreichend für 1M Timesteps Training
    """
    
    gamma: float = 0.99
    """Discount Factor - Standard für Atari"""
    
    tau: float = 1.0
    """Target Network Update Rate (1.0 = Hard Update)"""
    
    target_network_frequency: int = 1000
    """Target Network wird alle 1000 Steps aktualisiert"""
    
    batch_size: int = 32
    """Batch Size für Training"""
    
    # === EXPLORATION ===
    start_e: float = 1.0
    """Initiale Exploration Rate (100% zufällige Aktionen)"""
    
    end_e: float = 0.01
    """Finale Exploration Rate (1% zufällige Aktionen)"""
    
    exploration_fraction: float = 0.10
    """
    Anteil der Timesteps für Epsilon-Decay
    Bei 1M Steps: Epsilon sinkt von 1.0 auf 0.01 in 100K Steps
    """
    
    # === TRAINING ===
    learning_starts: int = 50_000
    """
    Warmup: Erste 50K Steps nur Daten sammeln, kein Training
    - Füllt Replay Buffer mit diversen Erfahrungen
    """
    
    train_frequency: int = 4
    """Trainiere alle 4 Environment Steps"""
    
    # === EVALUATION ===
    eval_frequency: int = 100_000
    """Evaluiere alle 100K Steps während des Trainings (bei 1M = 10 Evaluationen)"""
    
    eval_episodes: int = 100
    """Anzahl Episoden für finale Evaluation (wie in Aufgabe gefordert)"""
    
    eval_epsilon: float = 0.05
    """Epsilon während Evaluation (kleine Exploration für Robustheit)"""
    
    num_envs: int = 1
    """Anzahl paralleler Environments (DQN nutzt nur 1)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Erstellt eine Space Invaders Umgebung mit Standard-Wrappern.
    
    Environment Processing Pipeline:
    1. NoopResetEnv: Zufällige No-Op Aktionen am Start (bis 30)
    2. MaxAndSkipEnv: Frame Skipping (4 Frames) + Max über letzte 2 Frames
    3. EpisodicLifeEnv: Episode endet bei Leben-Verlust (für besseres Signal)
    4. FireResetEnv: Automatisch FIRE am Episoden-Start
    5. ClipRewardEnv: Rewards auf [-1, 0, 1] clippen
    6. ResizeObservation: 84x84 Pixel
    7. GrayScaleObservation: Graustufen
    8. FrameStack: 4 Frames gestackt (für Bewegungserkennung)
    
    Observation Space nach Processing: (4, 84, 84) - 4 gestackte Graustufenframes
    Action Space: Discrete(6) - NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Atari-spezifische Wrapper
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)  # Frame Skip = 4
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        
        # Observation Processing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    """
    Deep Q-Network Architektur (Nature DQN).
    
    Architektur:
    - Input: 4x84x84 (4 gestackte Graustufenframes)
    - Conv1: 32 Filter, 8x8, Stride 4 → 32x20x20
    - Conv2: 64 Filter, 4x4, Stride 2 → 64x9x9
    - Conv3: 64 Filter, 3x3, Stride 1 → 64x7x7
    - Flatten: 3136 Neuronen
    - FC1: 512 Neuronen
    - Output: 6 Neuronen (Q-Werte für jede Aktion)
    
    Aktivierung: ReLU
    Input-Normalisierung: x / 255.0 (Pixel von [0,255] auf [0,1])
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)  # Normalisierung auf [0, 1]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Lineare Epsilon-Decay Schedule."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_agent(model, env_id, seed, num_episodes, epsilon, device):
    """
    Konsistente Evaluation über num_episodes Episoden.
    
    Evaluations-Rezept (konsistent für alle Experimente):
    - Gleiche Environment-Wrapper wie im Training
    - Epsilon-greedy mit eval_epsilon für Robustheit
    - Sammelt: Rewards, Episode Lengths
    - Berechnet: Mean, Std, Min, Max, Median, Q25, Q75
    
    Returns:
        dict mit allen Metriken und Rohdaten
    """
    eval_env = gym.vector.SyncVectorEnv([
        make_env(env_id, seed, 0, False, "eval")
    ])
    
    model.eval()
    episode_rewards = []
    episode_lengths = []
    
    obs, _ = eval_env.reset(seed=seed)
    
    while len(episode_rewards) < num_episodes:
        if random.random() < epsilon:
            actions = np.array([eval_env.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        obs, rewards, terminations, truncations, infos = eval_env.step(actions)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_rewards.append(float(info["episode"]["r"]))
                    episode_lengths.append(int(info["episode"]["l"]))
    
    eval_env.close()
    model.train()
    
    rewards = np.array(episode_rewards[:num_episodes])
    lengths = np.array(episode_lengths[:num_episodes])
    
    return {
        "rewards": rewards.tolist(),
        "lengths": lengths.tolist(),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "q25": float(np.percentile(rewards, 25)),
        "q75": float(np.percentile(rewards, 75)),
        "mean_length": float(np.mean(lengths)),
    }


def save_results(args, eval_results, training_rewards, run_name, output_dir="results"):
    """Speichert alle Ergebnisse für späteren Vergleich."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "experiment": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": asdict(args),
        "final_evaluation": eval_results,
        "training_rewards": training_rewards,
    }
    
    filepath = os.path.join(output_dir, f"{args.exp_name}_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nErgebnisse gespeichert: {filepath}")
    return filepath


if __name__ == "__main__":
    import stable_baselines3 as sb3
    
    if sb3.__version__ < "2.0":
        raise ValueError("Bitte stable_baselines3 >= 2.0 installieren")
    
    args = tyro.cli(Args)
    assert args.num_envs == 1, "DQN unterstützt nur 1 Environment"
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y%m%d_%H%M%S')}"
    
    # TensorBoard Setup
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(args).items()])),
    )
    
    # Seeding für Reproduzierbarkeit
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Device Selection (CPU für M2 Mac, CUDA falls verfügbar)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"\n{'='*60}")
    print(f"DQN Initial Training - Space Invaders")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Trainingsbudget: {args.total_timesteps:,} Timesteps")
    print(f"Entspricht: {args.total_timesteps * 4:,} Frames (mit Frame Skip=4)")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # Environment Setup
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) 
        for i in range(args.num_envs)
    ])
    
    # Networks
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Replay Buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    
    # Training Tracking
    start_time = time.time()
    training_rewards = []  # Für Lernkurve
    intermediate_evals = []  # Zwischenevaluationen
    
    # Training Loop
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        # Epsilon-Greedy Action Selection
        epsilon = linear_schedule(
            args.start_e, args.end_e, 
            args.exploration_fraction * args.total_timesteps, 
            global_step
        )
        
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # Environment Step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Episode Logging
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_reward = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    training_rewards.append({
                        "step": global_step,
                        "reward": ep_reward,
                        "length": ep_length
                    })
                    print(f"Step {global_step:,} | Episode Reward: {ep_reward:.0f} | Length: {ep_length} | Epsilon: {epsilon:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_reward, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
        
        # Replay Buffer Update
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        
        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                
                # Logging
                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, global_step)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Target Network Update
            if global_step % args.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * q_param.data + (1.0 - args.tau) * target_param.data
                    )
        
        # Zwischenevaluation
        if global_step > 0 and global_step % args.eval_frequency == 0:
            print(f"\n--- Zwischenevaluation bei Step {global_step:,} ---")
            eval_result = evaluate_agent(
                q_network, args.env_id, args.seed + 1000, 
                20,  # Schnelle Zwischenevaluation mit 20 Episoden
                args.eval_epsilon, device
            )
            intermediate_evals.append({
                "step": global_step,
                "mean_reward": eval_result["mean"],
                "std_reward": eval_result["std"]
            })
            print(f"Mean Reward: {eval_result['mean']:.2f} ± {eval_result['std']:.2f}")
            writer.add_scalar("eval/intermediate_mean_reward", eval_result["mean"], global_step)
            print("--- Weiter mit Training ---\n")
    
    # Modell speichern
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"\nModell gespeichert: {model_path}")
    
    # Finale Evaluation (100 Episoden wie gefordert)
    print(f"\n{'='*60}")
    print(f"FINALE EVALUATION ({args.eval_episodes} Episoden)")
    print(f"{'='*60}")
    
    final_eval = evaluate_agent(
        q_network, args.env_id, args.seed + 2000,
        args.eval_episodes, args.eval_epsilon, device
    )
    
    print(f"\nERGEBNISSE DQN INITIAL:")
    print(f"  Mean Reward:   {final_eval['mean']:.2f}")
    print(f"  Std Reward:    {final_eval['std']:.2f}")
    print(f"  Min Reward:    {final_eval['min']:.0f}")
    print(f"  Max Reward:    {final_eval['max']:.0f}")
    print(f"  Median Reward: {final_eval['median']:.0f}")
    print(f"  Q25:           {final_eval['q25']:.0f}")
    print(f"  Q75:           {final_eval['q75']:.0f}")
    print(f"  Mean Length:   {final_eval['mean_length']:.0f}")
    
    # TensorBoard Logging
    writer.add_scalar("eval/final_mean_reward", final_eval["mean"], 0)
    writer.add_scalar("eval/final_std_reward", final_eval["std"], 0)
    writer.add_scalar("eval/final_min_reward", final_eval["min"], 0)
    writer.add_scalar("eval/final_max_reward", final_eval["max"], 0)
    
    # Ergebnisse speichern
    final_eval["intermediate_evals"] = intermediate_evals
    save_results(args, final_eval, training_rewards, run_name)
    
    # Trainingszeit
    total_time = time.time() - start_time
    print(f"\nTrainingszeit: {total_time/60:.1f} Minuten")
    print(f"Durchschnittliche SPS: {args.total_timesteps / total_time:.0f}")
    
    # Vergleich mit Baseline
    print(f"\n{'='*60}")
    print(f"VERGLEICH MIT BASELINE")
    print(f"{'='*60}")
    print(f"  Random Baseline Mean:  ~147")
    print(f"  DQN Initial Mean:      {final_eval['mean']:.2f}")
    improvement = ((final_eval['mean'] - 147) / 147) * 100
    print(f"  Verbesserung:          {improvement:+.1f}%")
    print(f"{'='*60}")
    
    envs.close()
    writer.close()
    
    print(f"\n{'='*60}")
    print("Training abgeschlossen!")
    print(f"TensorBoard: tensorboard --logdir runs")
    print(f"{'='*60}")
