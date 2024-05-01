import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT
import matplotlib.pyplot as plt

# define hyperparameters
LEARNING_RATE = 0.005  # 
DISCOUNT_FACTOR = 0.99
EPISODES = 10000
EPSILON_DECAY = 0.999  # controls how much explore-exploit ratio changes
MIN_EPSILON = 0.01

# define the Q-network
# super() gets the methods from nn.Module
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increased network capacity
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
        direction = game.direction

        state = [
            # food location
            (food_x - head_x) // GRID_WIDTH,
            (food_y - head_y) // GRID_HEIGHT,
            
            # nearby danger
            int(game.check_collision((head_x + LEFT[0], head_y + LEFT[1]))),
            int(game.check_collision((head_x + RIGHT[0], head_y + RIGHT[1]))),
            int(game.check_collision((head_x + UP[0], head_y + UP[1]))),
            int(game.check_collision((head_x + DOWN[0], head_y + DOWN[1]))),
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

def train_q_learning_agent():
    agent = Agent(6, 4)
    scores = []
    epsilon = 1.0
    
    
    for episode in range(EPISODES):
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
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            """  # Update the display
            game.screen.fill((0,0,0))
            game.draw_snake()
            game.draw_food()
            pygame.display.flip()
            game.clock.tick(10)  # Control the speed of the visualization

            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return """

        scores.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")
            """ keyboard = input("Press 'q' to quit, or 'Enter' to continue...")
            if keyboard.lower() == "q":
                print("\n\nUSER QUIT\n\n")
                pygame.quit()
                exit() """
        
        
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
    plt.savefig('plot.png')
