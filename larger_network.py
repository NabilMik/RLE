"""
RLE Mini-Challenge: Erweiterung 4 - Größeres Netzwerk für Space Invaders

ERWEITERUNG: Netzwerk-Architektur (Größeres Netzwerk)
=====================================================

HYPOTHESE:
----------
Ein größeres Netzwerk mit mehr Neuronen sollte komplexere Strategien lernen
können und möglicherweise bessere Rewards erreichen, da es mehr Kapazität
hat, um feine Unterschiede zwischen Zuständen zu erkennen.

ÄNDERUNG GEGENÜBER DQN INITIAL:
-------------------------------
| Layer          | DQN Initial | Größeres Netzwerk |
|----------------|-------------|-------------------|
| Conv1          | 32 Filter   | 32 Filter         |
| Conv2          | 64 Filter   | 64 Filter         |
| Conv3          | 64 Filter   | 128 Filter (+100%)|
| FC1            | 512 Neuronen| 1024 Neuronen (+100%)|
| FC2 (neu!)     | -           | 512 Neuronen (NEU)|
| Output         | 6 Neuronen  | 6 Neuronen        |

BEGRÜNDUNG:
-----------
- Mehr Filter in Conv3: Kann feinere visuelle Features erkennen
- Größere FC-Schichten: Mehr Kapazität für komplexe Entscheidungen
- Zusätzliche FC-Schicht: Tieferes Netzwerk kann abstraktere Repräsentationen lernen

RISIKO:
-------
- Größeres Netzwerk braucht mehr Daten zum Trainieren
- Könnte bei nur 1M Steps nicht konvergieren
- Höherer Speicherverbrauch

Alle anderen Hyperparameter bleiben IDENTISCH für fairen Vergleich.
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
    exp_name: str = "larger_network"
    """Name dieses Experiments - Größeres Netzwerk Erweiterung"""
    seed: int = 42
    """Random Seed für Reproduzierbarkeit"""
    torch_deterministic: bool = True
    cuda: bool = True
    
    # Logging & Speichern
    capture_video: bool = True
    save_model: bool = True
    
    # Environment
    env_id: str = "ALE/SpaceInvaders-v5"
    
    # === TRAININGSBUDGET (IDENTISCH ZU DQN INITIAL) ===
    total_timesteps: int = 1_000_000
    
    # === DQN HYPERPARAMETER (IDENTISCH ZU DQN INITIAL) ===
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    
    # === EXPLORATION (IDENTISCH ZU DQN INITIAL) ===
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    
    # === TRAINING (IDENTISCH ZU DQN INITIAL) ===
    learning_starts: int = 50_000
    train_frequency: int = 4
    
    # === EVALUATION ===
    eval_frequency: int = 100_000
    eval_episodes: int = 100
    eval_epsilon: float = 0.05
    
    # === CHECKPOINTS ===
    checkpoint_frequency: int = 250_000
    
    num_envs: int = 1


def make_env(env_id, seed, idx, capture_video, run_name):
    """Erstellt Environment mit Standard-Wrappern (identisch zu DQN Initial)."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        env.action_space.seed(seed)
        return env
    return thunk


# ============================================================
# GRÖSSERES NETZWERK - Das ist die Hauptänderung!
# ============================================================
class LargerQNetwork(nn.Module):
    """
    Größeres Q-Network mit mehr Kapazität.
    
    Änderungen gegenüber Standard DQN:
    - Conv3: 64 → 128 Filter (+100%)
    - FC1: 512 → 1024 Neuronen (+100%)
    - FC2: Neue Schicht mit 512 Neuronen
    
    Architektur:
    - Input: 4x84x84
    - Conv1: 32 Filter, 8x8, Stride 4 → 32x20x20
    - Conv2: 64 Filter, 4x4, Stride 2 → 64x9x9
    - Conv3: 128 Filter, 3x3, Stride 1 → 128x7x7 (GRÖSSER!)
    - Flatten: 6272 Neuronen (vorher 3136)
    - FC1: 1024 Neuronen (GRÖSSER!)
    - FC2: 512 Neuronen (NEU!)
    - Output: 6 Neuronen
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            # Convolutional Layers
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),  # 64 → 128 Filter!
            nn.ReLU(),
            nn.Flatten(),
            # Fully Connected Layers (größer!)
            nn.Linear(128 * 7 * 7, 1024),  # 512 → 1024!
            nn.ReLU(),
            nn.Linear(1024, 512),  # Neue Schicht!
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_agent(model, env_id, seed, num_episodes, epsilon, device):
    """Konsistente Evaluation (identisch zu DQN Initial)."""
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
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(args).items()])),
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"\n{'='*60}")
    print(f"GRÖSSERES NETZWERK Training - Space Invaders")
    print(f"{'='*60}")
    print(f"Erweiterung: Größere Netzwerk-Architektur")
    print(f"Änderungen vs. DQN Initial:")
    print(f"  - Conv3:  64 → 128 Filter (+100%)")
    print(f"  - FC1:    512 → 1024 Neuronen (+100%)")
    print(f"  - FC2:    NEU - 512 Neuronen (zusätzliche Schicht)")
    print(f"Device: {device}")
    print(f"Trainingsbudget: {args.total_timesteps:,} Timesteps")
    print(f"{'='*60}\n")
    
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) 
        for i in range(args.num_envs)
    ])
    
    # Größeres Netzwerk!
    q_network = LargerQNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = LargerQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Parameter-Anzahl ausgeben
    total_params = sum(p.numel() for p in q_network.parameters())
    print(f"Netzwerk-Parameter: {total_params:,} (Standard DQN: ~1.7M)")
    
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    training_rewards = []
    intermediate_evals = []
    
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
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
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
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
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        
        # Standard DQN Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                
                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, global_step)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if global_step % args.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * q_param.data + (1.0 - args.tau) * target_param.data
                    )
            
            # Checkpoint speichern
            if args.save_model and global_step % args.checkpoint_frequency == 0:
                checkpoint_path = f"runs/{run_name}/checkpoints/model_{global_step}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(q_network.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint gespeichert: {checkpoint_path}")
        
        if global_step > 0 and global_step % args.eval_frequency == 0:
            print(f"\n--- Zwischenevaluation bei Step {global_step:,} ---")
            eval_result = evaluate_agent(
                q_network, args.env_id, args.seed + 1000, 
                20, args.eval_epsilon, device
            )
            intermediate_evals.append({
                "step": global_step,
                "mean_reward": eval_result["mean"],
                "std_reward": eval_result["std"]
            })
            print(f"Mean Reward: {eval_result['mean']:.2f} ± {eval_result['std']:.2f}")
            writer.add_scalar("eval/intermediate_mean_reward", eval_result["mean"], global_step)
            print("--- Weiter mit Training ---\n")
    
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"\nModell gespeichert: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"FINALE EVALUATION ({args.eval_episodes} Episoden)")
    print(f"{'='*60}")
    
    final_eval = evaluate_agent(
        q_network, args.env_id, args.seed + 2000,
        args.eval_episodes, args.eval_epsilon, device
    )
    
    print(f"\nERGEBNISSE GRÖSSERES NETZWERK:")
    print(f"  Mean Reward:   {final_eval['mean']:.2f}")
    print(f"  Std Reward:    {final_eval['std']:.2f}")
    print(f"  Min Reward:    {final_eval['min']:.0f}")
    print(f"  Max Reward:    {final_eval['max']:.0f}")
    print(f"  Median Reward: {final_eval['median']:.0f}")
    print(f"  Q25:           {final_eval['q25']:.0f}")
    print(f"  Q75:           {final_eval['q75']:.0f}")
    print(f"  Mean Length:   {final_eval['mean_length']:.0f}")
    
    writer.add_scalar("eval/final_mean_reward", final_eval["mean"], 0)
    writer.add_scalar("eval/final_std_reward", final_eval["std"], 0)
    writer.add_scalar("eval/final_min_reward", final_eval["min"], 0)
    writer.add_scalar("eval/final_max_reward", final_eval["max"], 0)
    
    final_eval["intermediate_evals"] = intermediate_evals
    save_results(args, final_eval, training_rewards, run_name)
    
    total_time = time.time() - start_time
    print(f"\nTrainingszeit: {total_time/60:.1f} Minuten")
    print(f"Durchschnittliche SPS: {args.total_timesteps / total_time:.0f}")
    
    print(f"\n{'='*60}")
    print(f"VERGLEICH")
    print(f"{'='*60}")
    print(f"  Random Baseline Mean:  ~147")
    print(f"  DQN Initial Mean:      ~486")
    print(f"  Größeres Netzwerk:     {final_eval['mean']:.2f}")
    print(f"{'='*60}")
    
    envs.close()
    writer.close()
    
    print(f"\n{'='*60}")
    print("Größeres Netzwerk Training abgeschlossen!")
    print(f"TensorBoard: tensorboard --logdir runs")
    print(f"{'='*60}")
