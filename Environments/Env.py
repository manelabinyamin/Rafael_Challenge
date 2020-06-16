import numpy as np
import matplotlib.pyplot as plt
from box import Box
from Environments.Interceptor_V2 import Init, Draw, Game_step

class environment:
    def __init__(self):
        # game params
        self.max_step = 1000
        # env params
        self.min_city, self.max_city = -4600, 400
        self.min_angle, self.max_angle = -90, 90
        self.width, self.height = 10000, 4000  # [m]
        self.max_r_vel, self.max_i_vel = 1000, 800  # [m]
        self.action_space = 4
        state_space = {
            'rockets': (None,4),
            'interceptors': (None,4),
            'cities': (2,),
            'angle': (1,),
        }
        self.state_space = Box(state_space)

    def reset(self):
        self.step_count = 0
        self.done = False
        Init()
        self.all_rewards = []
        self.total_score = 0
        self.reward = 0
        r_locs, i_locs, c_locs, ang, _ = Game_step(1)
        # pre-process observation
        self.preprocess_observation(r_locs, i_locs, c_locs, ang)
        return self.obs

    def step(self, action):
        r_locs, i_locs, c_locs, ang, new_score = Game_step(action)
        # processs reward
        self.reward = new_score - self.total_score
        self.all_rewards.append(self.reward)
        self.total_score = new_score
        # pre-process observation
        self.preprocess_observation(r_locs, i_locs, c_locs, ang)
        # check if done
        self.step_count += 1
        if self.step_count >= self.max_step:
            self.done = True
        return self.obs, self.reward, self.done, None

    def render(self):
        Draw()

    def sample_action(self):
        return np.random.choice(self.action_space)

    def build_obs_dict(self):
        cities = np.array(self.c_locs)
        angle = np.array([self.angle])
        rockets = np.array(self.r_locs)
        interceptors = np.array(self.i_locs)
        obs = {
            'rockets': rockets[np.newaxis,:],
            'interceptors': interceptors[np.newaxis,:],
            'angle': angle[np.newaxis,:],
            'cities': cities[np.newaxis,:],
        }
        obs_vec = [obs['rockets'], obs['interceptors'], obs['cities'], obs['angle']]
        return obs_vec

    def preprocess_observation(self, r_locs, i_locs, c_locs, ang):
        self.c_locs = self.scale_cities(c_locs[:, 0])  # y is always 800
        self.angle = self.scale_angle(ang)
        self.i_locs = self.scale_missile(i_locs, type='interceptors') if len(i_locs)>0 else i_locs
        self.r_locs = self.scale_missile(r_locs, type='rockets') if len(r_locs)>0 else r_locs
        self.obs = self.build_obs_dict()

    def scale_cities(self, c_locs):
        '''
        scale location to the range [0,1]
        :param c_locs: pre-scaled location
        :return: scaled location
        '''
        return (np.array(c_locs) - self.min_city) / (self.max_city - self.min_city)

    def scale_angle(self, angle):
        '''
        scale angle to the range [0,1]
        :param angle: pre-scaled angle
        :return: scaled angle
        '''
        return (angle - self.min_angle) / (self.max_angle - self.min_angle)

    def scale_missile(self, missils, type='rockets'):
        '''
        scale missile location and velocity
        :param missils: pre-scaled location and velocity
        :return: scaled location and velocity
        '''
        assert type in ['rockets', 'interceptors']
        scaled_missils = []
        for m in missils:
            x, y = m[0]/self.width, m[1]/self.height
            if type == 'rockets':
                vx, vy = m[2]/self.max_r_vel, m[3]/self.max_r_vel
            else:
                vx, vy = m[2] / self.max_i_vel, m[3] / self.max_i_vel
            scaled_missils.append([x, y, vx, vy])
        return scaled_missils

    def get_state_space(self):
        return self.state_space.copy()

    def get_game_score(self):
        return self.total_score
