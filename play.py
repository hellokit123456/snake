# play.py
# Play a trained model (render) or play manually with arrow keys.
import pygame
import torch
import argparse
from snake_env import SnakeEnv
from dqn import QNetwork

def play(model_path=None, human=False):
    env = SnakeEnv(grid_size=10)
    obs = env.reset()
    if model_path:
        net = QNetwork(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()
    else:
        net = None

    running = True
    clock = pygame.time.Clock()
    while running:
        if human:
            # handle keyboard
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action = 0
                    elif event.key == pygame.K_RIGHT: action = 1
                    elif event.key == pygame.K_DOWN: action = 2
                    elif event.key == pygame.K_LEFT: action = 3
                    else: action = None
                    if action is not None:
                        obs, r, done, _ = env.step(action)
        else:
            # model action
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = net(s)
                action = int(q.argmax().item())
            obs, r, done, _ = env.step(action)

        env.render()
        if done:
            obs = env.reset()
        clock.tick(10)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--human", action='store_true')
    args = parser.parse_args()
    play(model_path=args.model, human=args.human)
