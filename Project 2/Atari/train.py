import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.optim as optim
import os
import random
from wrapper import PreprocessFrame, FrameStack
from model import DQN
from replay import ReplayMemory, Transition
import control

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return None
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.from_numpy(np.stack(batch.next_state)).float().to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_state_batch).gather(1, next_actions).squeeze(1)
    
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
    expected_q_values = expected_q_values.unsqueeze(1)

    loss = torch.nn.MSELoss()(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_DQN(env_name,
              num_episodes=100,
              replay_capacity=1_000,
              batch_size=8,
              gamma=0.99,
              learning_rate=1e-3,
              target_update=1000,
              initial_epsilon=1.0,
              final_epsilon=0.1,
              epsilon_decay=100_000,
              start_episode=0,
              steps_done=0,
              losses=None,
              episode_rewards=None,
              memory=None,
              resume_policy_net_state=None,
              resume_target_net_state=None):
    env = gym.make(env_name, render_mode='human')
    env = PreprocessFrame(env, shape=(84, 84))
    env = FrameStack(env, 4)
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    if resume_policy_net_state is not None:
        policy_net.load_state_dict(resume_policy_net_state)
    if resume_target_net_state is not None:
        target_net.load_state_dict(resume_target_net_state)

    if losses is None:
        losses = []
    if episode_rewards is None:
        episode_rewards = []
    if memory is None:
        memory = ReplayMemory(replay_capacity)
    
    for episode in range(start_episode, num_episodes):
        if control.stop_training:
            print("Training stopped by user.")
            break

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if control.stop_training:
                print("Stopping current episode due to stop signal.")
                done = True
                break

            env.unwrapped.render()

            epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-1. * steps_done / epsilon_decay)
            # epsilon = max(final_epsilon, initial_epsilon - (epsilon_decay * episode))

            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
                with torch.no_grad():
                    action = policy_net(state_tensor).max(1)[1].item()
            
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            loss = optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)
            if loss is not None:
                losses.append(loss)

            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            cv2.imshow("Control", control.control_image)
            cv2.waitKey(1)
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode:03d} | Reward: {int(total_reward):03d} | Epsilon: {epsilon:.3f}")

        if control.stop_training:
            checkpoint = {
                "episode": episode + 1,
                "steps_done": steps_done,
                "policy_net_state_dict": policy_net.state_dict(),
                "target_net_state_dict": target_net.state_dict(),
                "memory": memory,
                "losses": losses,
                "episode_rewards": episode_rewards,
            }
            torch.save(checkpoint, "DQN_checkpoint.pth")
            print("Checkpoint saved at episode", episode)
            break
    if not control.stop_training and os.path.exists("DQN_checkpoint.pth"):
        os.remove("DQN_checkpoint.pth")

    return policy_net, target_net, losses, episode_rewards
