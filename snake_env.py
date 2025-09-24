# snake_env.py
# Gym-style Snake environment (discrete grid)
# Observation: a flattened grid (0 empty, 1 snake, 2 food) OR you can swap to a compact observation.
# Actions: 0=up,1=right,2=down,3=left

import gym
from gym import spaces
import numpy as np
import random
import pygame
from typing import Tuple

# Config
GRID_SIZE = 10   # grid is GRID_SIZE x GRID_SIZE
CELL_PIX = 24    # pixels per cell for rendering
WINDOW_SIZE = GRID_SIZE * CELL_PIX

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        # action space
        self.action_space = spaces.Discrete(4)
        # observation: grid_size x grid_size integers in {0,1,2}
        self.observation_space = spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.int8)
        self.reset()

        # pygame renderer
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.score = 0
        self.done = False
        self.direction = random.choice([(0,-1),(1,0),(0,1),(-1,0)])  # (dx,dy)
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid-1, mid)]  # head first
        self._place_food()
        self.steps_since_food = 0
        self.max_steps_no_food = self.grid_size * self.grid_size * 2
        return self._get_obs()

    def _place_food(self):
        free = set((x,y) for x in range(self.grid_size) for y in range(self.grid_size)) - set(self.snake)
        self.food = random.choice(list(free))

    def _get_obs(self):
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for x,y in self.snake:
            g[y,x] = 1
        fx,fy = self.food
        g[fy,fx] = 2
        return g

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # map action to direction but disallow 180-degree reversals
        dirs = [(0,-1),(1,0),(0,1),(-1,0)]
        new_dir = dirs[action]
        # prevent reversal:
        if len(self.snake) > 1:
            head = self.snake[0]
            neck = self.snake[1]
            cur_dir = (head[0]-neck[0], head[1]-neck[1])
            if (new_dir[0] == -cur_dir[0] and new_dir[1] == -cur_dir[1]):
                new_dir = cur_dir
        self.direction = new_dir

        # move
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0

        # check collisions with walls
        x,y = new_head
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, True, {}

        # check collision with self
        if new_head in self.snake:
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, True, {}

        # insert head
        self.snake.insert(0, new_head)
        ate = (new_head == self.food)
        if ate:
            reward = 1.0
            self.score += 1
            self._place_food()
            self.steps_since_food = 0
        else:
            # pop tail
            self.snake.pop()
            self.steps_since_food += 1

        # optional small living penalty to encourage speed
        reward += -0.01

        # optional termination if starving
        if self.steps_since_food > self.max_steps_no_food:
            self.done = True
            reward = -1.0

        return self._get_obs(), reward, self.done, {}

    # Render either human (pygame window) or rgb array
    def render(self, mode='human'):
        obs = self._get_obs()
        if mode == 'rgb_array':
            surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            self._draw(surface, obs)
            return pygame.surfarray.array3d(surface).transpose([1,0,2])

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
        self._draw(self.window, obs)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw(self, surface, grid):
        surface.fill((0,0,0))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                val = grid[y,x]
                rect = pygame.Rect(x*CELL_PIX, y*CELL_PIX, CELL_PIX-1, CELL_PIX-1)
                if val == 1:
                    pygame.draw.rect(surface, (0,180,0), rect)
                elif val == 2:
                    pygame.draw.rect(surface, (200,50,50), rect)
                else:
                    pygame.draw.rect(surface, (40,40,40), rect)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
            self.clock = None
