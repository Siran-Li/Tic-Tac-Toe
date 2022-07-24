import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
from collections import namedtuple, deque
from tic_env import TictactoeEnv, OptimalPlayer
# from torch.optim.lr_scheduler import ExponentialLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 500
BUFFER_SIZE = 10000

EPS_MIN = 0.1
EPS_MAX = 0.8
steps_done = 0

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(18, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 9)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[q(x,a1),q(x,a2),...q(x,a9)]...]).
    def forward(self, x):
        x = x.view(x.size(0), -1).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


def select_action(policy_net, state, eps, decay=False, iter=None, nstar=None):
    # return [[action(0-8)]]
    global steps_done
    sample = random.random()

    if decay:
        eps_threshold = max(EPS_MIN, EPS_MAX*(1.0-(iter*1.0/nstar)))
    else:
        eps_threshold = eps

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(9)]], device=device, dtype=torch.long)




def optimize_model(policy_net, target_net, memory, transition, optimizer, MEM=True):
    # single step
    criterion = nn.SmoothL1Loss()
    if MEM:
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)  # batch x 3x3x2
        action_batch = torch.cat(batch.action)  # [batch,1]
        reward_batch = torch.cat(batch.reward)  # [batch] 


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        # print(state_action_values)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values =[]
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask =[True,False,True......]
        # non_final_next_states =[nexts1,nexts3,...]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)

        left_tensors = [s for s in batch.next_state if s is not None]
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if len(left_tensors) != 0:
            non_final_next_states = torch.cat(left_tensors)
            next_state_values[non_final_mask] = target_net(
                non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values

        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))
    else:
        state_action_values = policy_net(transition.state)
        if transition.next_state is not None:
            next_state_values = target_net(
                transition.state).detach().max().item()
        else:
            next_state_values = 0
        expected_state_action_values = (
            next_state_values * GAMMA) + transition.reward

        # Compute Huber loss
        loss = criterion(
            state_action_values[0, transition.action][0], expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()


def get_DQN_input(grid):
    # gets grid, return state for X
    grid_t = torch.tensor(grid)
    res = torch.zeros(3, 3, 2)
    res[:, :, 0] = torch.where(grid_t == 1, 1, 0)
    res[:, :, 1] = torch.where(grid_t == -1, 1, 0)
    return res


def policy_test(policy_net, player_opt):
    # Testing against some other player 'O'. return M.
    env = TictactoeEnv()
    first_player = 'X'
    pre_state = None
    rewards = []
    for i in range(500):
        torch.manual_seed(i)
        random.seed(i)
        np.random.seed(i)

        env.reset()
        if i >= 250:
            first_player = 'O'

        grid, _, __ = env.observe()
        if env.get_current_player() != first_player:
            env.change_current_player()

        for _ in range(9):
            if env.current_player == player_opt.player:
                move = player_opt.act(grid)
            else:
                # Player X Select and perform an action
                # Move to the next state
                pre_state = get_DQN_input(grid)[None, :]
                x_mov = select_action(policy_net, pre_state, eps=0)
                move = x_mov.item()

            grid, done, winner = env.step(move)

            if done:
                rewards.append(env.reward(player='X'))
                break

    wins = [r for r in rewards if r == 1]
    losses = [r for r in rewards if r == -1]
    return ((len(wins)-len(losses))/len(rewards))


def one_expert_train_game(env, player_opt, policy_net, target_net, optimizer, epsilon, decay, i_episode, nstar, batch, MEM):
    Turns = np.array(['X', 'O'])
    env.reset()
    first_player = Turns[i_episode % 2]

    grid, _, __ = env.observe()
    pre_state = None
    if env.get_current_player() != first_player:
        env.change_current_player()

    x_mov = None
    for t in range(9):
        if env.current_player == player_opt.player:
            move = player_opt.act(grid)
        else:
            # Player X Select and perform an action
            # Move to the next state
            pre_state = get_DQN_input(grid)[None, :]

            x_mov = select_action(
                policy_net, pre_state, eps=epsilon, decay=decay, iter=i_episode, nstar=nstar)
            move = x_mov.item()

        grid, done, winner = env.step(move)
        reward = torch.tensor([env.reward(player='X')], device=device)

        # Observe new state
        if ((env.current_player == 'O' and not done) or (pre_state is None)):
            # do nothing, wait for next O to play then observe next state and update
            continue

        if not done:
            # update after observing new state from player O
            next_state = get_DQN_input(grid)[None, :]
        else:
            next_state = None

        # Store the transition in memory
        transition = Transition(pre_state, x_mov, next_state, reward)
        if batch:
            MEM.push(pre_state, x_mov, next_state, reward)

        # Perform one step of the optimization (on the policy network)

        loss = optimize_model(policy_net, target_net, MEM,
                              transition, optimizer, batch)

        if done:
            return env.reward(player='X'), loss


def one_self_train_game(env, policy_net, target_net, optimizer, epsilon, decay, i_episode, nstar, MEM):
    # Initialize the environment and state
    Turns = np.array(['X', 'O'])
    env.reset()
    first_player = Turns[i_episode % 2]

    grid, _, __ = env.observe()
    pre_state_x = None
    pre_state_o = None
    if env.get_current_player() != first_player:
        env.change_current_player()

    x_mov = None
    o_mov = None
    for t in range(9):
        if env.current_player == 'X':
            pre_state_x = get_DQN_input(grid)[None, :]
            x_mov = select_action(
                policy_net, pre_state_x, eps=epsilon, decay=decay, iter=i_episode, nstar=nstar)
            move = x_mov.item()
        else:
            # Player O Select and perform an action
            # Move to the next state
            grid_o = np.where(grid == -0, 0, grid*(-1))
            pre_state_o = get_DQN_input(grid_o)[None, :]
            o_mov = select_action(
                policy_net, pre_state_o, eps=epsilon, decay=decay, iter=i_episode, nstar=nstar)
            move = o_mov.item()

        grid, done, winner = env.step(move)

        reward_x = torch.tensor([env.reward(player='X')], device=device)
        reward_o = torch.tensor([env.reward(player='O')], device=device)
        # O just finished, update X
        if ((env.current_player == 'X' and not done) and (pre_state_x is not None)):
            next_state_x = get_DQN_input(grid)[None, :]
            MEM.push(pre_state_x, x_mov, next_state_x, reward_x)
        # X just finished, update O
        elif((env.current_player == 'O' and not done) and (pre_state_o is not None)):
            grid_o = np.where(grid == -0, 0, grid*(-1))
            next_state_o = get_DQN_input(grid_o)[None, :]
            MEM.push(pre_state_o, o_mov, next_state_o, reward_o)
        # Game end, update both
        elif done:
            MEM.push(pre_state_x, x_mov, None, reward_x)
            MEM.push(pre_state_o, o_mov, None, reward_o)
        # first round no update
        else:
            continue
        # Perform one step of the optimization (on the policy network)
        loss = optimize_model(policy_net, target_net,
                              MEM, None, optimizer, True)

        if done:
            return env.reward(player='X'), loss


def play_games(env, num_episodes, epsilon=0.2, lr=5e-4, opts_epsilon=0.5, batch=True, decay=False, nstar=None, test=False, Print=False, self_play=False):
    # initialization
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    player_opt = OptimalPlayer(epsilon=opts_epsilon, player='O')
    M_opt = []
    M_rand = []

    rewards = np.zeros(num_episodes)
    training_loss = []
    MEM = None
    if batch:
        MEM = ReplayMemory(BUFFER_SIZE)
    # play num_episodes rounds of games
    for i_episode in tqdm(range(num_episodes)): 
        if self_play:
            reward, loss = one_self_train_game(
                env, policy_net, target_net, optimizer, epsilon, decay, i_episode, nstar, MEM)
        else:
            reward, loss = one_expert_train_game(
                env, player_opt, policy_net, target_net, optimizer, epsilon, decay, i_episode, nstar, batch, MEM)

        rewards[i_episode] = reward
        training_loss.append(loss)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if test and i_episode % 250 == 0:
            # test against opt(0) and opt(1)
            m_opt = policy_test(
                policy_net, OptimalPlayer(epsilon=0, player='O'))
            m_rand = policy_test(
                policy_net, OptimalPlayer(epsilon=1, player='O'))
            M_opt.append(m_opt)
            M_rand.append(m_rand)
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)


    return rewards, training_loss, M_opt, M_rand,policy_net
