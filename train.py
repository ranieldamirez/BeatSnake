import os
import torch
from tqdm import tqdm
from gameengine import SnakeGame
from agent import Agent

def train_agent(continue_training=False):
    agent = Agent(12, 4)
    model_path = 'q_network.pth'
    episodes = 2500
    epsilon_decay = 0.999
    min_epsilon = 0.01
    scores = []
    epsilon = 1.0

    if continue_training and os.path.exists(model_path):
        print("Loading existing model...")
        agent.q_network.load_state_dict(torch.load(model_path))

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

    torch.save(agent.q_network.state_dict(), model_path)
    print("Training completed and model saved.")

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
