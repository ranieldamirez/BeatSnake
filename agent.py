import numpy as np
import random
import pygame
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT

# Define hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9  # try 0.95 too
EPISODES = 10000

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Initialize pygame
pygame.init()

# Set screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = 20

# Initialize the font
font = pygame.font.Font(None, 36)

def state_to_index(state):
    x, y, food_x, food_y = state
    x = min(max(x, 0), GRID_WIDTH - 1)
    y = min(max(y, 0), GRID_HEIGHT - 1)
    food_x = min(max(food_x, 0), GRID_WIDTH - 1)
    food_y = min(max(food_y, 0), GRID_HEIGHT - 1)
    return y * GRID_WIDTH * GRID_WIDTH + x * GRID_WIDTH + food_y * GRID_WIDTH + food_x

class Agent:
    def __init__(self, state_space_size, action_space_size):
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, len(self.q_table[state]) - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        self.q_table[state, action] = new_q

def train_q_learning_agent():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Game')

    episode_count = 0
    running = True
    while running:
        # Create a new instance of the Snake game for each episode
        game = SnakeGame()

        state = (game.snake[0][0], game.snake[0][1], game.food[0], game.food[1])
        done = False
        total_reward = 0
        if episode_count % 500 == 0:
            print("\n ----- 500 CHECKPOINT ----- \n")
        if episode_count in range(9980, 10000):
            while not done:
                # Render the game
                screen.fill(BLACK)
                game.draw_grid()
                game.draw_snake()
                game.draw_food()
                pygame.display.flip()
                pygame.time.delay(100)

                # Choose action
                epsilon = 1 - (episode_count / EPISODES)  # Epsilon-greedy exploration strategy
                action = agent.choose_action(state_to_index(state), epsilon)

                # Take action
                reward, done = game.play_step(action)
                next_state = (game.snake[0][0], game.snake[0][1], game.food[0], game.food[1])

                # Update Q-table
                agent.update_q_table(state_to_index(state), action, reward, state_to_index(next_state))
                state = next_state
                total_reward += reward

            # Display total reward for the episode
            print(f"Episode Score: {total_reward}")

            # Increment episode count
            episode_count += 1

            # Wait for user to press Enter before starting next episode
            input("Press Enter to continue...")

        else:
            while not done:

                # Choose action
                epsilon = 1 - (episode_count / EPISODES)  # Epsilon-greedy exploration strategy
                action = agent.choose_action(state_to_index(state), epsilon)

                # Take action
                reward, done = game.play_step(action)
                next_state = (game.snake[0][0], game.snake[0][1], game.food[0], game.food[1])

                # Update Q-table
                agent.update_q_table(state_to_index(state), action, reward, state_to_index(next_state))
                state = next_state
                total_reward += reward

            # Display total reward for the episode
            print(f"Episode Score: {total_reward}")

            # Increment episode count
            episode_count += 1

            # Wait for user to press Enter before starting next episode
            #input("Press Enter to continue...")
        pygame.quit()

if __name__ == '__main__':
    agent = Agent(GRID_WIDTH * GRID_HEIGHT * GRID_WIDTH * GRID_HEIGHT, 4)
    train_q_learning_agent()
