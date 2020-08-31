# PPO code from LuEE-c's github repository:
# https://github.com/LuEE-C/PPO-Keras.git
# He took his initial framework from: https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from box import Box
import argparse
import neptune
import pprint as pp
from time import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Conv2D, Flatten, GlobalMaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
tf.compat.v1.disable_eager_execution()

import numba as nb

from Environments.Env import environment
from Environments.Env_train_cnn import environment as environment_train

""" -------Global Variables------- """
USE_NEPTUNE = True
LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
NUM_ACTIONS = 4
ENTROPY_LOSS = 1e-3
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


class Agent:
    def __init__(self, args):
        self.args = args
        self.train = args.train
        self.render_freq = args.render_freq
        self.render_ep = False
        self.env = environment_train() if self.train else environment()
        self.state_space = self.env.get_state_space()
        self.pooling_method = 'max'
        self.latent_space = 60
        head_network = self.build_head()
        self.critic = self.build_critic(head_network)
        self.actor = self.build_actor(head_network)
        plot_model(self.critic, to_file='critic_network.png', show_shapes=True)
        plot_model(self.actor, to_file='actor_network.png', show_shapes=True)
        plot_model(self.conv_layers(), to_file='cnn_head_network.png', show_shapes=True)

        # print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0
        self.avg_reward = []
        self.cur_rewards = []

    def build_head(self):
        """ Assemble shared layers
        """
        input_shape = self.state_space
        rockets = Input(shape=input_shape.image, name='rockets')
        interceptors = Input(shape=input_shape.image, name='interceptors')
        angle = Input(shape=input_shape.angle, name='angle')
        cities = Input(shape=input_shape.cities, name='cities')
        can_shoot = Input(shape=input_shape.can_shoot, name='can_shoot')

        cnn = self.conv_layers()
        combinedCNN = concatenate([cnn(rockets), cnn(interceptors)])
        combinedCNN = Dense(128, activation='relu')(combinedCNN)

        combinedInput = concatenate([combinedCNN, angle, cities, can_shoot])
        x = Dense(128, activation='relu')(combinedInput)
        # x = Dense(128, activation='relu')(x)
        model = Model(inputs=[rockets, interceptors, angle, cities, can_shoot], outputs=x)
        return model

    def conv_layers(self):
        """ Assemble conv layers
        """
        input = Input(shape=self.state_space.image, name='missiles')
        x = Conv2D(8, kernel_size=8, strides=2, padding='same',
                   activation='relu')(input)
        x = Conv2D(16, kernel_size=4, strides=2, padding='same',
                   activation='relu')(x)
        x = Conv2D(32, kernel_size=4, strides=2, padding='same',
                   activation='relu')(x)
        x = Conv2D(64, kernel_size=4, strides=2, padding='same',
                   activation='relu')(x)
        x = GlobalMaxPool2D()(x)
        out = Dense(128, activation='relu')(x)
        model = Model(input, out)
        return model

    def build_actor(self, head_network):
        x = Dense(128, activation='relu')(head_network.output)
        x = Dense(64, activation='relu')(x)
        out = Dense(NUM_ACTIONS, activation='softmax')(x)
        advantage = Input(shape=(1,), name='Advantage')
        old_prediction = Input(shape=(NUM_ACTIONS,), name='Old_prediction')
        model = Model(head_network.input+[advantage, old_prediction], out)

        model.compile(optimizer=Adam(lr=self.args.learning_rate),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self, head_network):
        x = Dense(128, activation='relu')(head_network.output)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='linear')(x)

        model = Model(head_network.input, out)
        model.compile(optimizer=Adam(lr=self.args.learning_rate), loss='mse')
        model.summary()
        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True  # play greedily
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []
        self.render_ep = True if self.render_freq>0 and self.episode%self.render_freq==0 else False

    def get_action(self):
        # find action
        p = self.actor.predict(self.observation+[DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def transform_reward(self):
        rewards = np.array([self.reward]).transpose()
        n = rewards.shape[0]
        cols,rows = np.meshgrid(range(n),range(n))
        factors = np.triu(np.power(self.args.gamma, cols-rows))
        self.reward = np.matmul(factors, rewards).tolist()

    def get_batch(self):
        batch = [[[],[],[],[],[]], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[1]) < self.args.buffer_size:
            if self.render_ep:
                self.env.render()
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done = self.env.step(action)
            if done:
                self.reward = reward

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.reward_over_time.append(self.env.total_rewards)
                self.cur_rewards.append(self.env.total_rewards)

                print('Episode: {} ; Score: {}'.format(self.episode, self.reward_over_time[-1]))
                if USE_NEPTUNE: neptune.log_metric('Rewards', self.reward_over_time[-1])

                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        [batch[0][k].append(obs[k]) for k in range(len(batch[0]))]
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                    # pad all lists to have the same length
                tmp_batch = [[], [], []]
                self.reset_env()

        batch[0] = [np.concatenate(x) for x in batch[0]]
        obs, action, pred, reward = batch[0], np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        if self.train:
            fig, ax = plt.subplots(1, 1)

            while self.episode < self.args.num_episodes:
                # ep_start_time = time()
                obs, action, pred, reward = self.get_batch()
                obs = [obs[k][:self.args.buffer_size] for k in range(len(obs))]
                action, pred, reward = action[:self.args.buffer_size], pred[:self.args.buffer_size], reward[:self.args.buffer_size]
                old_prediction = pred
                pred_values = self.critic.predict(obs)

                advantage = reward - pred_values
                actor_loss = self.actor.fit(x=obs+[advantage, old_prediction], y=action, batch_size=self.args.batch_size, shuffle=True, epochs=self.args.epochs, verbose=False)
                critic_loss = self.critic.fit(obs, reward, batch_size=self.args.batch_size, shuffle=True, epochs=self.args.epochs, verbose=False)
                if USE_NEPTUNE:
                    neptune.log_metric('Actor loss', actor_loss.history['loss'][-1])
                    neptune.log_metric('Critic loss', critic_loss.history['loss'][-1])

                self.gradient_steps += 1

                self.avg_reward.append(np.mean(self.cur_rewards))
                self.cur_rewards = []

                # plot results
                if len(ax.lines) > 1:
                    ax.lines[-1].remove()
                ax.plot(range(len(self.avg_reward)), self.avg_reward, color='blue')
                ax.axhline(max(self.avg_reward), linestyle='--', color='red', linewidth=0.5)
                ax.set_title('Score over episode')
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.0001)
        else:  # test
            pass


def main(args):
    if USE_NEPTUNE:
        neptune.init('manelab/Rafael-challenge')
        neptune.create_experiment(tags=['Keras','PPO','CNN'],params=args)
    ag = Agent(Box(args))
    ag.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for the agent')

    # agent parameters
    parser.add_argument('--train', help='Do you want to train the agent (False = test)', type=bool, default=True)
    parser.add_argument('--render_freq', help='render frequency (number of episodes). If 0, the environment will not be rendered',
                        type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1e6)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gamma', type=int, default=0.95)
    parser.add_argument('--buffer_size', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=1e-4)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

