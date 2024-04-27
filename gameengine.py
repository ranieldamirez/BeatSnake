import pygame
import random

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set screen dimensions
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
CELL_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Define directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('SNAKE')
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.food = self.generate_food()
        self.last_snake_length = len(self.snake)

    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (SCREEN_WIDTH, y))

    def draw_snake(self):
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_food(self):
        pygame.draw.rect(self.screen, RED, (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def move_snake(self):
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)
        if head == self.food:
            self.food = self.generate_food()
        else:
            self.snake.pop()

    def play_step(self, action):
        # Perform the action in the game
        if action == 0:
            self.direction = UP
        elif action == 1:
            self.direction = DOWN
        elif action == 2:
            self.direction = LEFT
        elif action == 3:
            self.direction = RIGHT

        # Move the snake
        self.move_snake()

        # Check if the game is done after taking the action
        done = self.check_collision()

        # Calculate reward based on game state
        reward = 0
        if done:
            reward = -1  # Negative reward for losing the game
        elif len(self.snake) > self.last_snake_length:
            reward = 1  # Positive reward for growing the snake
        else:
            reward = 0  # No reward for other steps

        self.last_snake_length = len(self.snake)

        return reward, done

    def check_collision(self):
        head = self.snake[0]
        if head[0] < 0 or head[0] >= GRID_WIDTH or head[1] < 0 or head[1] >= GRID_HEIGHT or head in self.snake[1:]:
            return True
        return False

    def run_game(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake()
            self.draw_food()
            pygame.display.flip()
            self.clock.tick(30000)

        pygame.quit()
