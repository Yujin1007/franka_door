import numpy as np
import torch
from torch.nn import Module, Linear, LSTM, ReLU, Sequential, LayerNorm
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid
from gym import spaces
import gym
import pickle
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data


from tqc import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class RescaleAction(gym.ActionWrapper):
    # def __init__(self, env, a, b):
    #     assert isinstance(env.action_space, spaces.Box), (
    #         "expected Box action space, got {}".format(type(env.action_space)))
    #     assert np.less_equal(a, b).all(), (a, b)
    #     super(RescaleAction, self).__init__(env)
    #     self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
    #     self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
    #     self.action_space = spaces.Box(low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype)
    def __init__(self, env):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        super(RescaleAction, self).__init__(env)
        self.a = env.action_space.low
        self.b = env.action_space.high
        self.action_space = env.action_space

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*((action - self.a)/(self.b - self.a))
        action = np.clip(action, low, high)
        return action


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

class LSTM_MLP(Module):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim, num_layers, mlp_hidden_dims, output_dim, IsDemo=False):
        super(LSTM_MLP, self).__init__()
        self.lstm = LSTM(state_dim, lstm_hidden_dim, num_layers, batch_first=True)
        if IsDemo:
            self.mlp = Normalized_MLP(lstm_hidden_dim + action_dim, mlp_hidden_dims, output_dim)
        else:
            self.mlp = MLP(lstm_hidden_dim+action_dim, mlp_hidden_dims, output_dim)

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers

    def forward(self, state, action):
        batch_size = state.size(0)
        if state.ndim == 2:
            state = state.reshape(1, state.shape[0], state.shape[1])
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(state.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(state.device)
        _, (hn, _) = self.lstm(state, (h0, c0))
        lstm_output = hn[-1]
        x = torch.cat((lstm_output, action), dim=1)
        return self.mlp(x)


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

class Normalized_MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Normalized_MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(Linear(input_dim, hidden_dim))
            layers.append(LayerNorm(hidden_dim))
            layers.append(ReLU())
            input_dim = hidden_dim
        layers.append(Linear(input_dim, output_dim))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# class Mlp(Module):
#     def __init__(
#             self,
#             input_size,
#             hidden_sizes,
#             output_size
#     ):
#         super().__init__()
#         # TODO: initialization
#         self.fcs = []
#         in_size = input_size
#         for i, next_size in enumerate(hidden_sizes):
#             fc = Linear(in_size, next_size)
#             self.add_module(f'fc{i}', fc)
#             self.fcs.append(fc)
#             in_size = next_size
#         self.last_fc = Linear(in_size, output_size)
#
#     def forward(self, input):
#         h = input
#         for fc in self.fcs:
#             h = relu(fc(h))
#         output = self.last_fc(h)
#         return output
#

# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.transition_names = ('state', 'action', 'next_state', 'reward', 'not_done')
#         sizes = (state_dim, action_dim, state_dim, 1, 1)
#         for name, size in zip(self.transition_names, sizes):
#             if type(size) is int:
#                 setattr(self, name, np.empty((max_size, size)))
#             elif type(size) is tuple:
#                 setattr(self, name, np.empty((max_size, size[0], size[1])))
#             else:
#                 raise Exception('유효한 dimension type이 아님')
#
#     def add(self, state, action, next_state, reward, done):
#         values = (state, action, next_state, reward, 1. - done)
#         for name, value in zip(self.transition_names, values):
#             getattr(self, name)[self.ptr] = value
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#         names = self.transition_names
#         return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)
#
#     def get_all(self):
#         return {name: getattr(self, name)[:self.size] for name in self.transition_names}
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(DEVICE)

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
        x = ReLU(x)
        x = self.conv2(x, self.edge_index)
        return x

class GCN_MLP(Module):
    def __init__(self, state_dim, action_dim, gnn_hidden_dim, mlp_hidden_dims, output_dim, edge_index):
        super(GCN_MLP, self).__init__()
        self.gnn = GCN(state_dim, gnn_hidden_dim, edge_index)
        self.mlp = MLP(gnn_hidden_dim+action_dim, mlp_hidden_dims, output_dim)
        self.hidden_dim = gnn_hidden_dim

    def forward(self, state, action):
        if state.ndim == 2:
            state = state.reshape(1, state.shape[0], state.shape[1])
        gnn_output = self.gnn.forward(state)
        x = torch.cat((gnn_output, action), dim=1)
        return self.mlp(x)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.transition_names = ('state', 'action', 'next_state', 'reward', 'not_done')
        sizes = (state_dim, action_dim, state_dim, 1, 1)
        for name, size in zip(self.transition_names, sizes):
            if type(size) is int:
                setattr(self, name, np.empty((max_size, size)))
            elif type(size) is tuple:
                setattr(self, name, np.empty((max_size, size[0], size[1])))
            else:
                raise Exception('Invalid dimension type')

    def add(self, state, action, next_state, reward, done):
        values = (state, action, next_state, reward, 1. - done)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # if self.ptr == self.max_size:
            # print("buffer full")

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        names = self.transition_names
        return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)

    def get_all(self):
        return {name: getattr(self, name)[:self.size] for name in self.transition_names}

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

class Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets, IsDemo=False):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        lstm_hidden_dim = 256

        for i in range(n_nets):
            net = LSTM_MLP(state_dim, action_dim, lstm_hidden_dim=lstm_hidden_dim, num_layers=1, mlp_hidden_dims=[512, 512, 512], output_dim=n_quantiles,IsDemo=IsDemo)

            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        # sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(state,action) for net in self.nets), dim=1)
        return quantiles


class GNN_Actor(Module):
    def __init__(self, state_dim, action_dim, edge_index):
        super().__init__()
        self.action_dim = action_dim
        gnn_hidden_dim = 256
        self.gnn_feature_extractor = GCN(state_dim, gnn_hidden_dim, edge_index)
        self.net = MLP(gnn_hidden_dim, [256, 256], 2 * action_dim)

    def forward(self, obs):
        feature = self.gnn_feature_extractor(obs)
        mean, log_std = self.net(feature).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        '''custom normalized activation'''
        # action = L2NormActivation()(action)
        return action, log_prob

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action

class GNN_Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets,edge_index):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        gnn_hidden_dim = 256

        for i in range(n_nets):
            net = GCN_MLP(state_dim, action_dim, gnn_hidden_dim,[512, 512, 512], n_quantiles, edge_index)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        # sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(state,action) for net in self.nets), dim=1)
        return quantiles


class Actor(Module):
    def __init__(self, state_dim, action_dim, IsDemo=False):
        super().__init__()
        self.action_dim = action_dim
        lstm_hidden_dim = 256
        self.lstm_feature_extractor = LSTMFeatureExtractor(state_dim, hidden_dim=lstm_hidden_dim, num_layers=1)
        if IsDemo:
            self.net = Normalized_MLP(lstm_hidden_dim, [256, 256], 2 * action_dim)
        else:
            self.net = MLP(lstm_hidden_dim, [256, 256], 2 * action_dim)

    def forward(self, obs):
        feature = self.lstm_feature_extractor(obs)
        mean, log_std = self.net(feature).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        '''custom normalized activation'''
        # action = L2NormActivation()(action)
        return action, log_prob

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action


class L2NormActivation(Module):
    def __init__(self):
        super(L2NormActivation, self).__init__()

    def forward(self, x):
        # Split the output into head and tail parts
        head_output = x[:, :3]
        tail_output = x[:, 3:]

        # Calculate L2 norm for the head and tail parts
        head_norm = torch.norm(head_output, p=2, dim=1, keepdim=True)
        tail_norm = torch.norm(tail_output, p=2, dim=1, keepdim=True)

        # Normalize the head and tail parts based on L2 norms
        head_output = head_output / head_norm
        tail_output = tail_output / tail_norm

        # Concatenate the normalized head and tail parts
        normalized_output = torch.cat([head_output, tail_output], dim=1)
        return normalized_output

class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
                                      torch.ones_like(self.normal_std, device=DEVICE))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh
