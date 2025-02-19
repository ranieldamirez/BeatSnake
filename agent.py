"""
# agent.py

This script is the composition of the AI Agent using a Q-Learning architecture.
It also plots the rewards gained by the model in each Snake game as it learns to maximize rewards.
"""

import numpy as np
from termcolor import colored
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT
import matplotlib
matplotlib.use('Agg')
from statistics import mean # DEBUG

# hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99

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
        target_tensor = torch.tensor(target_q_value, device = q_value.device)
        loss = nn.MSELoss()(q_value, target_tensor) # mean squared error to calculate loss between calculated q-value and ideal q-value

        self.optimizer.zero_grad() # clear previous gradients
        loss.backward() # backpropagation to compute gradients
        self.optimizer.step() # use gradients to adjust NN values
        return loss.item()
