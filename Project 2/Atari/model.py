import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 24, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2),
            nn.ReLU()
        )

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        conv_width  = conv2d_size_out(conv2d_size_out(input_shape[1], 6, 3), 4, 2)
        conv_height = conv2d_size_out(conv2d_size_out(input_shape[2], 6, 3), 4, 2)
        linear_input_size = conv_width * conv_height * 48

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)
