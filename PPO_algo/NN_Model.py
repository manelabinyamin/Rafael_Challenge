import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, nn_args, device):
        super(ActorCritic, self).__init__()
        self.nn_args = nn_args
        self.device = device
        pools = {'avg': pool_avg, 'max': pool_max, 'sum': pool_sum}
        self.pool = pools[self.nn_args.pooling_method]
        self.DS_encoder = nn.Linear(4,self.nn_args.latent_space)

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(self.args.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.args.action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(self.args.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def DeepSets(self, seq):
        embed_seq = []
        for s in seq:
            s = torch.tensor(s).resize_(1, len(s)).to(self.device)
            embed_seq.append(self.DS_encoder(s))
        embed_sequence = torch.cat(embed_seq, 0).unsqueeze(0).to(self.device)
        encoded = self.pool(embed_sequence, dim=1)
        return encoded

    def pad_seq(self, seq):
        lens = list(map(lambda x: x.shape[1], seq)) if len(seq)>0 else [0]
        max_len = max(max(lens),1)
        padded_seq = np.zeros([len(seq), max_len, 4])
        mask = np.arange(max_len) < np.array(lens)[:, None]
        mask = mask[:,:,np.newaxis]
        padded_seq[np.repeat(mask,4, axis=-1)] = np.concatenate(list(map(lambda x: x.flatten(),seq)))
        return padded_seq, mask

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)


def pool_sum(tensor, dim):
    return torch.sum(tensor, dim)
