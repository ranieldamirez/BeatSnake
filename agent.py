"""
# agent.py

This script is the composition of the AI Agent using a Q-Learning architecture.
It also plots the loss of the model throughout the training session.
"""

import numpy as np
from termcolor import colored
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean # DEBUG

# define hyperparameters
LEARNING_RATE = 0.001  # 
DISCOUNT_FACTOR = 0.99
EPISODES = 2500
EPSILON_DECAY = 0.999  # controls how much explore-exploit ratio changes
MIN_EPSILON = 0.01
collided = False

# define the Q-network
# super() gets the methods from nn.Module
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
    
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_space_size, action_space_size):
        self.q_network = QNetwork(state_space_size, action_space_size)
        # Adam optimization algorithm (similar to stochasitc gradient descent)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

    def get_state(self, game):
        head_x, head_y = game.snake[0]
        food_x, food_y = game.food

        # add direction
        dir_left = game.direction == LEFT
        dir_right = game.direction == RIGHT
        dir_up = game.direction == UP
        dir_down = game.direction == DOWN

        state = [ 
            # nearby danger
            int(game.check_collision((head_x + LEFT[0], head_y + LEFT[1]))),
            int(game.check_collision((head_x + RIGHT[0], head_y + RIGHT[1]))),
            int(game.check_collision((head_x + UP[0], head_y + UP[1]))),
            int(game.check_collision((head_x + DOWN[0], head_y + DOWN[1]))),

            # direction in which the snake is moving
            dir_left, dir_right, dir_up, dir_down,

            # food location
            food_x > head_x,  # food right
            food_x < head_x,  # Food left
            food_y > head_y,  # Food down
            food_y < head_y   # Food up
        ]

        return torch.tensor([state], dtype=torch.float)

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad(): # not doing inference here
                q_values = self.q_network(state)
                return torch.argmax(q_values).item() # return highest value as integer

    def update_q_network(self, state, action, reward, next_state, done):

        q_values = self.q_network(state) # give value [[q_value_action1, q_value_action2, ...]]
        next_q_values = self.q_network(next_state)

        max_next_q_value = torch.max(next_q_values).item()
        target_q_value = reward + DISCOUNT_FACTOR * max_next_q_value # bellman equation, discount future rewards and prioritize short term rewards

        q_value = q_values[0, action]
        loss = nn.MSELoss()(q_value, torch.tensor(target_q_value)) # mean squared error to calculate loss between calculated q-value and ideal q-value

        self.optimizer.zero_grad() # clear previous gradients
        loss.backward() # backpropagation to compute gradients
        self.optimizer.step() # use gradients to adjust NN values
        return loss.item()

def train_q_learning_agent():
    agent = Agent(12, 4)
    scores = []
    snake_lengths = []
    losses = []
    epsilon = 1.0
    
    for episode in tqdm(range(EPISODES), desc="Training Progress"):
        #print("\n EPISODE:", episode)
        episode_losses = []

        total_reward = 0
        game = SnakeGame()
        state = agent.get_state(game)
        done = False
        translation = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT"}
        
        while not done:
            pygame.event.pump()  # Process event queue
            action = agent.choose_action(state, epsilon)
            reward, done = game.play_step(action)
            next_state = agent.get_state(game)
            loss = agent.update_q_network(state, action, reward, next_state, done)
            episode_losses.append(loss)
            state = next_state
            total_reward += reward
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        average_loss = sum(episode_losses) / len(episode_losses)
        snake_lengths.append(len(game.snake))
        
        losses.append(average_loss)

        scores.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}: Score = {sum(scores)}, Avg. Snake Length = {sum(snake_lengths) / len(snake_lengths)}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Average Loss per Episode')
    plt.title("Loss During Training")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig('loss.png')
    
    return scores

if __name__ == '__main__':
    scores = train_q_learning_agent()
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
