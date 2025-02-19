"""
# train.py

This script runs the training session for the model, collecting the loss over time and tracking the
training progress via the tqdm library.
"""

import os
import torch
from tqdm import tqdm
from gameengine import SnakeGame
from agent import Agent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def train_agent(continue_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(12, 4)
    agent.q_network.to(device) # Move model to GPU if available
    model_path = 'q_network.pth'
    episodes = 2500
    epsilon_decay = 0.999
    min_epsilon = 0.01
    scores = []
    loss = []
    epsilon = 1.0

    if continue_training and os.path.exists(model_path):
        print("Loading existing model...")
        agent.q_network.load_state_dict(torch.load(model_path))

    for episode in tqdm(range(episodes), desc="Training Progress"):
        total_reward = 0
        game = SnakeGame()
        state = agent.get_state(game).to(device)
        done = False
        episode_loss = 0
        while not done:
            action = agent.choose_action(state, epsilon)
            reward, done = game.play_step(action)
            next_state = agent.get_state(game).to(device)
            episode_loss += agent.update_q_network(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        loss.append(episode_loss)
        scores.append(total_reward)

    torch.save(agent.q_network.state_dict(), model_path)
    print("Training completed and model saved.")

    plt.figure(figsize=(10, 5))
    plt.plot(loss, label='Average Loss per Episode')
    plt.title("Loss During Training")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig('loss.png')

    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Episode Rewards')
    z = np.polyfit(range(len(scores)), scores, 1)
    p = np.poly1d(z)
    plt.plot(range(len(scores)), p(range(len(scores))), "r--", label='Trendline')
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig('scores.png')

def main():
    model_path = 'q_network.pth'
    if os.path.exists(model_path):
        user_choice = input("Model found. Would you like to re-train from scratch (R) or continue training (C)? [R/C]: ").strip().upper()
        continue_training = (user_choice == 'C')
    else:
        continue_training = False
    
    train_agent(continue_training=continue_training)

if __name__ == '__main__':
    main()