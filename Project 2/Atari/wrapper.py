import gymnasium as gym
import cv2
import numpy as np
from collections import deque

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84), grayscale=True):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape
        self.grayscale = grayscale
        channels = 1 if self.grayscale else 3
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(channels, *self.shape), dtype=np.uint8)
        
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        
        if self.grayscale:
            obs = np.expand_dims(obs, axis=0)
        
        return obs
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0]
        
        return self.observation(obs)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(k, shp[-2], shp[-1]), dtype=env.observation_space.dtype)
    
    def reset(self):
        result = self.env.reset()
        ob = result[0]

        self.frames.clear()
        frame = np.array(ob, dtype=np.uint8)
        if frame.shape[0] == 1:
            frame = np.squeeze(frame, axis=0)   # (height, weight)
        for _ in range(self.k):
            self.frames.append(frame)   
        
        return np.stack(self.frames, axis=0)    # (k, height, width)

    def step(self, action):
        result = self.env.step(action)
        ob, reward, terminated, truncated, info = result
        done = terminated or truncated

        frame = np.array(ob, dtype=np.uint8)
        if frame.shape[0] == 1:
            frame = np.squeeze(frame, axis=0)
        self.frames.append(frame)

        return np.stack(self.frames, axis=0), reward, done, info
    

