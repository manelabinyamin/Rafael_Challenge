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

        self.head_network = nn.Sequential(
            nn.Linear(self.nn_args.hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        # actor
        self.action_layer = nn.Sequential(
            self.head_network,
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.nn_args.action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            self.head_network,
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


    def forward(self):
        raise NotImplementedError

    def DeepSets(self, input, mask):
        input = torch.from_numpy(input).float().to(self.device)
        mask = torch.from_numpy(mask).float().to(self.device)
        encoded = self.DS_encoder(input)
        encoded = encoded * mask
        out = self.pool(encoded, dim=1)
        return out

    # def DeepSets(self, seq):
    #     if len(seq[0]) > 0:
    #         embed_seq = []
    #         for s in seq:
    #             s = torch.from_numpy(s).float().to(self.device)
    #             encoded = self.DS_encoder(s)
    #             encoded = self.pool(encoded, dim=0)
    #             embed_seq.append(encoded)
    #         encoded_sequence = torch.cat(embed_seq, 0).unsqueeze(0).to(self.device)
    #
    #     else:
    #         encoded_sequence = torch.tensor([[0.0]*self.nn_args.latent_space]).to(self.device)
    #     return encoded_sequence

    def act(self, state, memory):
        rockets, interceptors, cities, angle = state
        rockets, mask_r = pad_seq(rockets)
        interceptors, mask_i = pad_seq(interceptors)
        encod_rockets, encod_interceptors = self.DeepSets(rockets, mask_r), self.DeepSets(interceptors, mask_i)
        cities, angle = torch.from_numpy(cities).float().to(self.device), torch.from_numpy(angle).float().to(self.device)
        cat_input = torch.cat([encod_rockets,encod_interceptors,cities,angle], 1)
        action_probs = self.action_layer(cat_input)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states_rockets.append(state[0])
        memory.states_interceptors.append(state[1])
        memory.states_cities.append(state[2])
        memory.states_angles.append(state[3])
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        rockets, interceptors, cities, angle = state
        rockets, mask_r = pad_seq(rockets)
        interceptors, mask_i = pad_seq(interceptors)
        encod_rockets, encod_interceptors = self.DeepSets(rockets, mask_r), self.DeepSets(interceptors, mask_i)
        # cities, angle = torch.tensor(cities).float().to(self.device), torch.tensor(angle).float().to(self.device)
        cities, angle = np.squeeze(np.array(cities),axis=1), np.squeeze(np.array(angle),axis=1)
        cities, angle = torch.from_numpy(np.array(cities)).float().to(self.device), torch.from_numpy(np.array(angle)).float().to(self.device)
        cat_input = torch.cat([encod_rockets, encod_interceptors, cities, angle], -1)

        action_probs = self.action_layer(cat_input)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(cat_input)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)


def pool_sum(tensor, dim):
    return torch.sum(tensor, dim)


def pad_seq(seq):
    if isinstance(seq, list):
        lens = list(map(lambda x: x.shape[1], seq)) if len(seq) > 0 else [0]
    else:
        lens = list(map(lambda x: x.size//4, seq)) if seq.size>0 else [0]
    max_len = max(max(lens),1)
    padded_seq = np.zeros([len(seq), max_len, 4])
    mask = np.arange(max_len) < np.array(lens)[:, None]
    mask = mask[:,:,np.newaxis]
    padded_seq[np.repeat(mask,4, axis=-1)] = np.concatenate(list(map(lambda x: x.flatten(),seq)))
    return padded_seq, mask
