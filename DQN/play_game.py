import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym
import cv2
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
import sys
import time
np.bool8 = np.bool_ 

GAME = 'CartPole-v1'
FPS = 50
frame_time = 1.0 / FPS  

env = gym.make(GAME, render_mode='rgb_array')

# creating a seperate env for visualization
env_human = gym.make(GAME, render_mode='human')

obs = env.reset()
obs_human = env_human.reset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

# -------------------------------------Model------------------------------------- #

file_path = str(sys.argv[1])
NUM_ACTIONS = 2

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        # layers may be tweaked depending on the task
        self.conv1 = nn.Conv2d(3, 6, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.fc1 = nn.Linear(16 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_ACTIONS) # cart pole has 2 possible actions per state

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DQN().to(device)

# loading the trained model
checkpoint = torch.load(file_path)
model.load_state_dict(checkpoint['model_state_dict'])

# -----------------------------------Functions----------------------------------- #

def preprocess_frame(frame):
    
    """"
    Preprocess an RGB frame for DQN input.

    Args:
        frame (np.ndarray): Input RGB frame as a NumPy array.

    Returns:
        torch.Tensor: Preprocessed frame resized to (84, 84) and normalized to [0, 1].
    """
    # resizing the frame and normalizing it
    frame_resized = cv2.resize(frame, (84, 84))
    frame_normalized = frame_resized.astype(np.float32) / 255

    # seperating each channel
    r = frame_normalized[:, :, 0]
    g = frame_normalized[:, :, 1]
    b = frame_normalized[:, :, 2]

    # converting to tensor
    preprocessed_frame = torch.tensor(np.array([r, g, b]), dtype=torch.float32)

    return preprocessed_frame

def compute_Q(states, NN):

    """
    Perform a forward pass through the NN to compute the Q values.

    Args:
        states (torch.Tensor): A tensor of states (batch_size, state_size).
        NN (torch.nn.Module): The neural network model used to compute the Q-values.

    Returns:
        torch.Tensor: The computed Q-values (batch_size, num_actions).
    """

    Q_values = NN.forward(states)

    return Q_values

def choose_action(state, NN):

    """
    Chooses an action greedily, for a given state.

    Args:
        NN (torch.nn.Module): The neural network model used to compute the Q-values.

    Returns:
        int: The action with the highest Q value.
    """

    # obtaining the Q predictions
    state_batch = state.unsqueeze(0).to(device)
    Q_prediction = NN.forward(state_batch)

    # obtaining the action from the Q predictions
    action = torch.argmax(Q_prediction) 

    # converting action to an int
    action = int(action) 

    return action

# -----------------------------------Game Loop----------------------------------- #

num_attempts = int(sys.argv[2])

print('Game is starting...')
for attempt_num in range(num_attempts):

    # resetting gameIsRunning
    gameIsRunning=True

    # reseting the environment at the start of each game loop
    env.reset()
    env_human.reset()

    # starting the game loop
    while gameIsRunning:

        # rendering both environments
        frame = env.render()
        env_human.render()
        
        # process the frame into the appropriate format
        state = preprocess_frame(frame)

        # choosing optimal action
        action = choose_action(state, model)

        # taking the step
        obs, reward, terminated, truncated, info = env.step(action)
        env_human.step(action)

        # stopping the game loop when necessary
        gameIsRunning = not (terminated or truncated)

        # sleeping to maintain 50 FPS
        time.sleep(frame_time)

print('Done playing...')