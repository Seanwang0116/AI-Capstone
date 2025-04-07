import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size), mode='valid') / window_size

def plot_rewards_and_losses(losses, eval_rewards, save_path, rolling_length=10):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    smoothed_losses = moving_average(losses, rolling_length)
    axs[0].plot(smoothed_losses, label="Smoothed Training Loss", color='blue')
    axs[0].set_title("Smoothed Training Loss over Time")
    axs[0].set_xlabel("Training Steps")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    smoothed_rewards = moving_average(eval_rewards, rolling_length)
    axs[1].plot(smoothed_rewards, label="Smoothed Evaluation Rewards", color='green')
    axs[1].set_title("Smoothed Evaluation Episode Rewards")
    axs[1].set_xlabel("Evaluation Episodes")
    axs[1].set_ylabel("Total Reward")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_policy(policy_net, losses, episode_rewards, env, num_episodes=3, device=torch.device("cpu"), render=False):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        if render:
            env.unwrapped.render()
        
        while not done:
            state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).max(1)[1].item()
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.unwrapped.render()
                time.sleep(0.02)
            
            state = next_state
        
        rewards.append(total_reward)
        print(f"Evaluation Episode {episode} | Reward: {total_reward}")

    save_path = "result_exponential.jpg"
    plot_rewards_and_losses(losses, episode_rewards, save_path, rolling_length=10)
    
    if render:
        env.unwrapped.close()
    
