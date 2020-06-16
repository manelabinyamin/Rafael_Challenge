import torch
import torch.nn as nn
from torch.distributions import Categorical
from PPO_algo.PPO import PPO, Memory
from Environments.Env import environment
from time import time
import matplotlib.pyplot as plt
import box
import numpy as np


def get_params():
    net_params = {
        'pooling_method': 'max',
        'latent_space': 50,
        'hidden_dim': 103,
        'action_dim': 4
    }
    return box.Box(net_params)


def plot_scores(ax, scores, add_legend=False):
    ax.plot(range(len(scores)), scores, color='blue', label='Score')
    ax.set_title('Game Score')
    if add_legend:
        ax.legend()


def main():
    ############## Hyperparameters ##############
    # plots
    fig, ax = plt.subplots(1, 1)
    plot_scores(ax, [], add_legend=True)
    # creating environment
    env = environment()
    render = False
    solved_reward = 100  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    lr = 1e-4
    entropy_loss = 1e-3
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    nn_params = get_params()
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    memory = Memory()
    ppo = PPO(nn_params, lr, betas, gamma, K_epochs, eps_clip)

    # training loop
    ep_num = 0
    training_scores = []
    smoothed_scores = []
    iter = 0
    max_score = -10000
    try:
        while ep_num < max_episodes:
            iter += 1
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
            if len(smoothed_scores) > 5:
                smoothed_scores.append(0.2*training_scores[-1]+0.8*smoothed_scores[-1])
            else:  # first five scores
                smoothed_scores.append(np.mean(training_scores))

            # update agent on the last k episodes
            ppo.update(memory)

            # Logging
            print('Iteration {} | Iter-time: {} | Score {} |'.format(iter, time()-start_time, training_scores[-1]))
            plot_scores(ax, smoothed_scores)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.0001)
            if training_scores[-1] > max_score:
                # save model
                max_score = training_scores[-1]
                #TODO: save model

    except KeyboardInterrupt:
        # save model, scores and smoothing scores
        raise


if __name__ == '__main__':
    main()


