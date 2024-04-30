import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is non-interactive and does not require a windowing system
import matplotlib.pyplot as plt
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT

# Define hyperparameters
LEARNING_RATE = 0.15  # Slightly increased
DISCOUNT_FACTOR = 0.95  # Slightly increased for longer-term planning
EPISODES = 100000
EPSILON_DECAY = 0.995  # More gradual decay
MIN_EPSILON = 0.01

class Agent:
    def __init__(self):
        self.q_table = np.zeros((9 * 4 * 2, 4))

    def get_state(self, game):
        head_x, head_y = game.snake[0]
        food_x, food_y = game.food

        food_dir_x = np.sign(food_x - head_x)
        food_dir_y = np.sign(food_y - head_y)
        food_dir_idx = (food_dir_x + 1) * 3 + (food_dir_y + 1)

        danger = [0] * 3
        directions = [LEFT, game.direction, RIGHT]
        for index, dir in enumerate(directions):
            next_pos = (head_x + dir[0], head_y + dir[1])
            if next_pos in game.snake or next_pos[0] < 0 or next_pos[0] >= GRID_WIDTH or next_pos[1] < 0 or next_pos[1] >= GRID_HEIGHT:
                danger[index] = 1

        state = (food_dir_idx, 2 * danger[0] + danger[1], danger[2])
        return np.ravel_multi_index(state, (9, 4, 2))

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        self.q_table[state, action] = new_q

def train_q_learning_agent():
    agent = Agent()
    scores = []
    epsilon = 1.0

    for episode in range(EPISODES):
        game = SnakeGame()
        state = agent.get_state(game)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            reward, done = game.play_step(action)
            next_state = agent.get_state(game)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        scores.append(total_reward)
        # Define colors for terminal output
        RED = "\033[91m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        if episode % 100 == 0:
            if total_reward < 0:
                color = RED
            elif total_reward == 0:
                color = YELLOW
            else:
                color = GREEN
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {color}{total_reward}{RESET}")

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

if __name__ == '__main__':
    train_q_learning_agent()
