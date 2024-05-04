import torch
from tqdm import tqdm
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def train_agent():
    agent = Agent(12, 4)
    episodes = 2500
    epsilon_decay = 0.999
    min_epsilon = 0.01
    scores = []
    epsilon = 1.0

    for episode in tqdm(range(episodes), desc="Training Progress"):
        total_reward = 0
        game = SnakeGame()
        state = agent.get_state(game)
        done = False
        
        while not done:
            action = agent.choose_action(state, epsilon)
            reward, done = game.play_step(action)
            next_state = agent.get_state(game)
            agent.update_q_network(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        scores.append(total_reward)

    torch.save(agent.q_network.state_dict(), 'q_network.pth')

if __name__ == '__main__':
    train_agent()
