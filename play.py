import pygame
import torch
from gameengine import SnakeGame, UP, DOWN, LEFT, RIGHT, GRID_WIDTH, GRID_HEIGHT
from agent import Agent

def play():
    agent = Agent(12, 4)
    agent.q_network.load_state_dict(torch.load('q_network.pth'))
    agent.q_network.eval()

    game = SnakeGame()
    state = agent.get_state(game)
    done = False

    while not done:
        action = agent.choose_action(state, 0)  # epsilon = 0 for no randomness
        _, done = game.play_step(action)
        state = agent.get_state(game)

        game.screen.fill((0,0,0))
        game.draw_snake()
        game.draw_food()
        pygame.display.flip()

        game.clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        

    pygame.quit()
    
    print(f"\nScore: {game.SCORE}\n")
if __name__ == '__main__':
    play()
