import torch.nn as nn


class Network(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes):
        super(Network, self).__init__()

        layers = []
        pre = n_states
        for ls in layers_sizes:
            pos = ls
            ll = nn.Linear(pre, pos)
            nn.init.xavier_uniform_(ll.weight)
            layers.append(ll)
            layers.append(nn.ReLU())

            pre = pos
        layers.append(nn.Linear(pre, n_actions))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes, cuda):
        super(DQN, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        device = 'cuda' if cuda else 'cpu'

        self.net = Network(n_states, n_actions, layers_sizes).to(device)
        self.target = Network(n_states, n_actions, layers_sizes).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()