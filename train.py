# train.py
import numpy as np
import torch
from snake_env import SnakeEnv
from dqn import DQNAgent
from tqdm import trange
import argparse
import os

def train(num_episodes=2000, save_path="models", device='cpu'):
    env = SnakeEnv(grid_size=10)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions, device=device, lr=1e-3)
    os.makedirs(save_path, exist_ok=True)

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 0.995

    epsilon = epsilon_start

    for ep in trange(1, num_episodes+1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.train_step(batch_size=64)
            state = next_state
            episode_reward += reward
            steps += 1

        # decay epsilon
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

        if ep % 50 == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(save_path, f"dqn_{ep}.pth"))
        if ep % 10 == 0:
            print(f"Episode {ep} reward {episode_reward:.2f} epsilon {epsilon:.3f}")

    # save final
    torch.save(agent.policy_net.state_dict(), os.path.join(save_path, "dqn_final.pth"))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(num_episodes=args.episodes, device=args.device)
