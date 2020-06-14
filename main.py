import torch
import torch.nn as nn
from torch.distributions import Categorical
from PPO_algo.PPO import PPO, Memory
from Environments.Env import environment
from time import time
import matplotlib.pyplot as plt



def plot_scores(ax, scores, add_legend=False):
    ax.plot(range(len(scores)), scores, color='blue', label='Score')
    ax.set_title('Game Score')
    if add_legend:
        ax.legend()


def main():
    ############## Hyperparameters ##############
    # plots
    fig, axes = plt.subplots(1, 1)
    plot_scores(axes[0], [], add_legend=True)
    # creating environment
    env = environment()
    state_dim = env.state_space
    action_dim = env.action_space
    render = False
    solved_reward = 30  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 100000  # max training episodes
    max_timesteps = 1000  # max timesteps in one episode
    lr = 1e-4
    entropy_loss = 1e-3
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, entropy_loss, betas, gamma, K_epochs, eps_clip)

    # training loop
    ep_num = 0
    training_scores = []
    iter = 0
    max_score = -10000
    while ep_num < max_episodes:
        start_time = time()
        temp_scores = 0
        memory.clear_memory()
        # play k episodes
        for e in range(K_epochs):
            ep_num += 1
            # play one episode
            state = env.reset()
            done = False
            while not done:
                # Running policy_old:
                action = ppo.policy_old.act(state, memory)
                state, reward, done, _ = env.step(action)

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

            temp_scores += env.get_game_score()

        # save scores
        training_scores.append(float(temp_scores/K_epochs))

        # update agent on the last k episodes
        ppo.update(memory)

        # Logging
        print('Iteration {} | Iter-time: {} | Score {} |'.format(iter, time()-start_time, training_scores[-1]))
        plot_scores(axes[0], training_scores)
        if training_scores[-1] > max_score:
            # save model
            max_score = training_scores[-1]
            #TODO: save model


if __name__ == '__main__':
    main()
