import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym
import cv2
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
import sys
np.bool8 = np.bool_ 

GAME = 'CartPole-v1'

env = gym.make(GAME, render_mode='rgb_array')

# creating a seperate env for visualization
env_human = gym.make(GAME, render_mode='human')

obs = env.reset()
obs_human = env_human.reset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

# -------------------------------------Model------------------------------------- #

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

online_NN = DQN()
target_NN = DQN()
online_NN.to(device)
target_NN.to(device)

if len(sys.argv) > 1:
    file_path = str(sys.argv[1]) 
    checkpoint = torch.load(file_path)
    online_NN.load_state_dict(checkpoint['model_state_dict'])
    target_NN.load_state_dict(checkpoint['model_state_dict'])

loss_fn = nn.MSELoss()  
optimizer = AdamW(online_NN.parameters(), lr=1e-4, weight_decay=0.01) 

# ---------------------- Dataset and Preprocessing Functions ---------------------- #

class DQNDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.replay_buffer[idx]
        return state.cpu(), \
               torch.tensor(action, dtype=torch.int64).cpu(), \
               torch.tensor(reward, dtype=torch.float32).cpu(), \
               next_state.cpu(), \
               torch.tensor(done, dtype=torch.int64).cpu()

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

# -------------------------------Training Functions------------------------------- #

def collect_experience(env, epsilon, env_human=None):
        
    """
    Collect a single transition using an epsilon-greedy policy.

    With probability `epsilon`, a random action is selected, otherwise, the action with the highest Q-value is chosen.
    The function returns the state, action, reward, next state, and done flag.

    Args:
        env (gym.Env): The environment instance in rgb mode.
        env_human (gym.Env, optional): The environment instance in human mode. If `None`, the GUI will not be rendered. Default is `None`.
        epsilon (float): The probability of selecting a random action.

    Returns:
        tuple: (state, action, reward, next_state, done), where:
            - state (torch.Tensor): The current state.
            - action (int): The selected action.
            - reward (float): The reward received.
            - next_state (torch.Tensor): The next state.
            - done (bool): True if the episode is done (either due to termination or truncation).
    """

    # obtaining the initial state
    frame = env.render()
    if env_human:
        env_human.render()
    state = preprocess_frame(frame)

    # obtaining the Q predictions
    state_batch = state.unsqueeze(0).to(device)
    Q_prediction = online_NN.forward(state_batch)

    # choose a random action with probability epsilon
    rand_num = np.random.uniform(0, 1)
    if rand_num < epsilon:
        action = env.action_space.sample()
    else:
        action = torch.argmax(Q_prediction) 

    # converting action to an int to pass into env.step()
    action = int(action) 

    # taking the step
    obs, reward, terminated, truncated, info = env.step(action)
    if env_human:
        env_human.step(action)

    # obtaining the next state
    frame = env.render()
    if env_human:
        env_human.render()
    next_state = preprocess_frame(frame)

    # store 1 if the state is a terminal or truncted
    done = int(terminated or truncated)

    return (state, action, reward, next_state, done)


def fill_replay_buffer(init_buffer_size, env, epsilon):

    """
    Fill the replay buffer with random or epsilon-greedy experience.

    Args:
        init_buffer_size (int): Number of transitions to fill the buffer with initially.
        env (gym.Env): The environment instance in rgb mode.
        epsilon (float): Probability of choosing a random action (exploration).

    Returns:
        list: List of (state, action, reward, next_state, done) transitions.
    """

    replay_buffer = []

    for i in range(init_buffer_size):
        
        # collecting an experience
        experience = collect_experience(env, epsilon)

        # storing the experience in the replay buffer
        replay_buffer.append(experience)

        # reseting if state is truncated or terminated
        if experience[-1]:
            env.reset()

    return replay_buffer
    

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


def compute_target_Q(batch, target_NN, gamma):

    """
    Compute the target Q-values using the Bellman equation.

    Args:
        batch (tuple): A tuple containing states, actions, rewards, next_states, and done flags.
        target_NN (nn.Module): The target neural network.
        gamma (float): The discount factor.

    Returns:
        torch.Tensor: The target Q-values for each state-action pair in the batch.
    """

    # unpacking values
    states, actions, rewards, next_states, done = batch

    # moving tensors to the gpu (if there is one)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    done = done.to(device)

    # computing target Q
    next_Q_values = compute_Q(next_states, target_NN)
    target_Q_values = rewards + (gamma * torch.max(next_Q_values, dim=1)[0] * (1 - done))

    return target_Q_values


def sgd_step(batch, target_Q, device):

    """
    Perform one SGD step on a sampled batch from the replay buffer.

    Args:
        batch (tuple): A batch of (states, actions, rewards, next_states, dones).
        device (torch.device): Device to run computations on.

    Returns:
        float: The loss for this step.
    """

    # unpacking values
    states, actions, rewards, next_states, done = batch

    # moving tensors to the gpu (if there is one)
    states = states.to(device)
    actions = actions.to(device)

    # computing the Q values from the batch
    predicted_Q_values = compute_Q(states, online_NN)
    Q_predictions_of_actions = predicted_Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # calculating the loss
    loss = loss_fn(Q_predictions_of_actions, target_Q.detach())

    # back propagating and updating weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def prepare_val_set(val_set_size, env):

    """
    Prepares the validation dataset by filling the replay buffer with initial experiences.

    Args:
        val_set_size (int): The number of experiences to collect for validation
        env (gym.Env): The environment instance in rgb mode.

    Returns:
        DQNDataset: the dataset containing the experiences for validation
    """

    # reusing fill_replay_buffer
    val_list = fill_replay_buffer(val_set_size, env, 0)
   
    return DQNDataset(val_list)

def validate(val_dataloader):

    """
    Compute the average Q-value of the validation set.

    Args:
        val_dataloader (torch.utils.data.DataLoader): The DataLoader object providing the validation set in batches.

    Returns:
        float: The average Q-value across the validation set.
    """

    total_max_Q = 0
    for batch_idx, batch in enumerate(val_dataloader):
        
        # unpacking the experiences
        states, actions, rewards, next_states, done = batch

        # moving states to device
        states = states.to(device)

        # obtaining Q values
        Q_values_batch = compute_Q(states, online_NN)

        # obtaining the max Q
        max_Q = Q_values_batch.max(dim=1).values 

        # add the sum of the batch Q values to total_Q
        total_max_Q += max_Q.sum().item()

    # recovering val_set_size
    val_set_size = len(val_dataloader.dataset)

    return total_max_Q / val_set_size 


# ------------------------------------Training------------------------------------ #

M = 10000 # the total number of episodes
T = 10000 # the maximum number of actions per episode
N = 4 # number of experiences to collect before performing SGD
C = 1 # the number of episodes between updating the target NN
UPDATES_BETWEEN_VAL = 1000 # the number of weight updates between model validation
VAL_SET_SIZE = 10000 # the size of the validation set
EPSILON = 1 # the proability that a random move will be selected
MIN_EPSILON = 0.1 # the minimum epsilon attainable
EPSILON_DECAY_FACTOR = 0.99977 # the value epsilon is multiplied by after each episode
GAMMA = 0.95 # the priority placed on future rewards
BATCH_SIZE = 32
INIT_BUFFER_SIZE = int(10000 / 2) # number of experiences to fill the buffer with initially
MAX_BUFFER_SIZE = 10000 # the maximum number of experiences the buffer can hold
MODEL_NAME = 'DQN_cartpole'

# filling the replay buffer and validation set
print('Filling replay buffer and validation set...')
replay_buffer = fill_replay_buffer(INIT_BUFFER_SIZE, env, EPSILON)
val_set = prepare_val_set(VAL_SET_SIZE, env)
val_set_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

# initializing num_updates to keep track of the number of times the model weights are updated
num_updates = 0

print('Starting training...')
for episode_number in range(M):

    # resetting the start_new_ep flag
    start_new_ep = False

    if (episode_number + 1) % 100 == 0:
        print(f'     Episode {episode_number + 1}: ')
    
    # reseting the environment at the start of each episode
    env.reset()
    env_human.reset()

    # updating target_NN weights
    if num_updates % C == 0:
        target_NN.load_state_dict(online_NN.state_dict())

    # starting a new episode if game is not terminated in T actions
    for t in range(T):

        # collecting N new experiences to update the buffer
        new_experiences = []
        for i in range(N):

            # collecting a new experience
            experience = collect_experience(env, EPSILON, env_human=env_human)
            new_experiences.append(experience)

            # ending the episode if the new state is terminal
            if experience[-1] == 1:
                # print('     Terminal state reached. Starting a new episode...')
                start_new_ep = True
                break

        # start a new episode if the flag is set
        if start_new_ep:
            break  

        # adding N new experiences to the buffer
        replay_buffer.extend(new_experiences)

        # ensuring the buffer size does not exceed MAX_BUFFER_SIZE
        if len(replay_buffer) > MAX_BUFFER_SIZE:
            replay_buffer = replay_buffer[-MAX_BUFFER_SIZE:]

        # preparing the replay buffer for sampling
        replay_buffer_prepared = DQNDataset(replay_buffer)
        sampler = RandomSampler(replay_buffer_prepared, replacement=True, num_samples=BATCH_SIZE)

        # sampling from the replay experience
        samples_dataloader = DataLoader(replay_buffer_prepared, batch_size=BATCH_SIZE, sampler=sampler)

        # extracting the sample
        batch = next(iter(samples_dataloader)) # there is only 1 batch 

        # computing target Q
        target_Q_values = compute_target_Q(batch, target_NN, GAMMA)

        # performing SGD
        loss = sgd_step(batch, target_Q_values, device)

        # validating
        if num_updates % UPDATES_BETWEEN_VAL == 0:
            print('Validating...')
            avg_Q = validate(val_set_dataloader)
            print(f'     Average Q of the validation set: {avg_Q}')

        

    # reducing EPSILON as training progresses
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY_FACTOR)

    # incrementing num_updates
    num_updates += 1

# saving the model weights
torch.save({'model_state_dict': online_NN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f"/Users/tanxe/Programming/ML/DQN/models/{MODEL_NAME}.pt")
    
env.close()
env_human.close()
