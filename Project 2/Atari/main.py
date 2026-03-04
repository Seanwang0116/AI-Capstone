import os
import torch
import cv2
import gymnasium as gym
from train import train_DQN
from wrapper import PreprocessFrame, FrameStack
from evaluate import evaluate_policy
from model import DQN

def Train_agent(save_path):
    check_point_path = "DQN_checkpoint.pth"
    if os.path.exists(check_point_path):
        checkpoint = torch.load(check_point_path, map_location="cpu", weights_only=False)
        start_episode = checkpoint["episode"]
        steps_done = checkpoint["steps_done"]
        losses = checkpoint["losses"]
        episode_rewards = checkpoint["episode_rewards"]
        memory = checkpoint["memory"]
        resume_policy_net_state = checkpoint["policy_net_state_dict"]
        resume_target_net_state = checkpoint["target_net_state_dict"]
        print(f"Resuming training from episode {start_episode}")
    else:
        start_episode = 0
        steps_done = 0
        losses = []
        episode_rewards = []
        memory = None
        resume_policy_net_state = None
        resume_target_net_state = None

    policy_net, target_net, losses, episode_rewards = train_DQN(
        env,
        num_episodes=200,
        target_update=1000,
        epsilon_decay=50_000,
        start_episode=start_episode,
        steps_done=steps_done,
        losses=losses,
        episode_rewards=episode_rewards,
        memory=memory,
        resume_policy_net_state=resume_policy_net_state,
        resume_target_net_state=resume_target_net_state,
    )

    saved = {   
        "losses": losses,
        "episode_rewards": episode_rewards,
        "policy_net_state_dict": policy_net.state_dict(),
    }
    torch.save(saved, save_path)
    cv2.destroyWindow("Control")

def Eval_agent(save_path):
    render_env = gym.make(env)
    render_env = PreprocessFrame(render_env, shape=(84, 84))
    render_env = FrameStack(render_env, 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if os.path.exists(save_path):
        saved = torch.load(save_path, map_location="cpu", weights_only=False)
        losses = saved["losses"]
        episode_rewards = saved["episode_rewards"]
        num_actions = render_env.action_space.n
        input_shape = render_env.observation_space.shape
        policy_net = DQN(input_shape, num_actions).to(device)
        policy_net.load_state_dict(saved["policy_net_state_dict"])

        evaluate_policy(policy_net, losses, episode_rewards, render_env, num_episodes=3, device=device, render=True)
    else:
        print("Not train yet!")

if __name__ == '__main__':
    import ale_py
    Train = True
    Eval = not Train

    env = "ALE/Assault-v5"
    verification_env = gym.make(env)
    print("Action meanings:", verification_env.unwrapped.get_action_meanings())
    verification_env.close()

    save_path = "DQN_assault_exponential.pth"
    if Train:
        Train_agent(save_path)
    else:
        Eval_agent(save_path)
