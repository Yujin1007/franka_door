import numpy as np
import torch
from torch.nn import Module, Linear, LSTM, ReLU, Sequential, LayerNorm
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid
from gym import spaces
import gym
import pickle
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(Linear(input_dim, hidden_dim))
            layers.append(ReLU())
            input_dim = hidden_dim
        layers.append(Linear(input_dim, output_dim))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LSTMFeatureExtractor(Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):

        if x.ndim == 2:
            x = x.reshape(1,x.shape[0], x.shape[1])
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return hn[-1]


class BC_Actor(Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        lstm_hidden_dim = 256
        self.lstm_feature_extractor = LSTMFeatureExtractor(state_dim, hidden_dim=lstm_hidden_dim, num_layers=1)
        self.net = MLP(lstm_hidden_dim, [256, 256], action_dim)

    def forward(self, obs):
        feature = self.lstm_feature_extractor(obs)
        action = self.net(feature)

        return action

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
        action = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action


class ExpertDataset(object):
    def __init__(self, data_path1, data_path2):
        with open(data_path1, 'rb') as f:
            data1 = pickle.load(f)
        with open(data_path2, 'rb') as f:
            data2 = pickle.load(f)
        data_len = int(data1.state.shape[0]*0.5)
        self.observations = data1.state[:data_len]
        self.actions = np.concatenate([data1.action[:data_len], data2.action[:data_len]], axis=1)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        action = self.actions[idx]
        return torch.tensor(observation, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


