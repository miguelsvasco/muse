import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DDPG(nn.Module):
    def __init__(self, n_states, n_actions, hidden1, hidden2, cuda):
        super(DDPG, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        if cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        self.actor = Actor(n_states, n_actions, hidden1, hidden2).to(device)
        self.actor_target = Actor(n_states, n_actions, hidden1,
                                  hidden2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic(self.n_states, self.n_actions, hidden1,
                             hidden2).to(device)
        self.critic_target = Critic(self.n_states, self.n_actions, hidden1,
                                    hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())