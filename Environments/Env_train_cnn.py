import numpy as np
import matplotlib.pyplot as plt
from box import Box
from Environments.Interceptor_V2_train_cnn import Init, Draw, Game_step, Get_rewards


class environment():
    def __init__(self):
        # game params
        self.stack_num = 3
        self.image_shape = [100, 80]
        self.max_step = 1000
        # env params
        self.min_city, self.max_city = -4600, 400
        self.min_angle, self.max_angle = -90, 90
        self.width, self.height = 10000, 4000  # [m]
        self.resolution = 100
        self.action_space = 4
        state_space = {
            # 'image': np.array([self.height//self.resolution, self.width//self.resolution, self.stack_num]),
            'image': (self.height//self.resolution, self.width//self.resolution, self.stack_num,),
            'cities': (2,),
            'angle': (1,),
            'can_shoot': (1,)
        }
        self.state_space = Box(state_space)
        self.interception_threshold = 2000
        self.real_interception = 150
        self.decay_rate = 0.9
        self.score_target = 100
        self.last_shoot = -10
        self.can_shoot = 1.0

    def reset(self):
        self.step_count = 0
        self.last_shoot = -10
        self.can_shoot = 1.0
        self.done = False
        self.r_stack, self.i_stack = None, None
        Init()
        self.all_rewards = []
        self.total_rewards = 0
        self.reward = 0
        r_locs, i_locs, c_locs, ang, _ = Game_step(1)
        # pre-process observation
        self.preprocess_observation(r_locs, i_locs, c_locs, ang)
        return self.obs

    def step(self, action):
        r_locs, i_locs, c_locs, ang, score = Game_step(action)
        # update can_shoot
        if self.can_shoot and action == 3:
            self.last_shoot = self.step_count
            self.can_shoot = 0.0
        else:
            if self.step_count - self.last_shoot >= 7:  # reload time is 7 steps
                self.can_shoot = 1.0
        # process reward
        self.reward = score - self.total_rewards
        self.all_rewards.append(self.reward)
        self.total_rewards = score
        # pre-process observation
        self.preprocess_observation(r_locs, i_locs, c_locs, ang)
        # check if done
        self.step_count += 1
        if self.step_count >= self.max_step:
            self.done = True
            self.reward = Get_rewards()
        return self.obs, self.reward, self.done

    def render(self):
        Draw()
        # plt.imshow(self.i_stack*100)
        # plt.show()

    def sample_action(self):
        return np.random.choice(self.action_space)

    def build_obs_dict(self):

        cities = np.array(self.c_locs)
        angle = np.array([self.angle])
        can_shoot = np.array([self.can_shoot])
        obs = {
            'rockets': self.r_stack[np.newaxis,:,:,:],
            'interceptors': self.i_stack[np.newaxis,:,:,:],
            'angle': angle[np.newaxis,:],
            'cities': cities[np.newaxis,:],
            'can_shoot': can_shoot[np.newaxis,:]
        }
        obs_vec = [obs['rockets'], obs['interceptors'], obs['angle'], obs['cities'], obs['can_shoot'].astype(float)]
        return obs_vec

    def preprocess_observation(self, r_locs, i_locs, c_locs, ang):
        self.c_locs = self.scale_cities(c_locs[:, 0])  # y is always 800
        self.angle = self.scale_angle(ang)
        r_img, i_img = self.location_to_img(r_locs), self.location_to_img(i_locs)
        self.r_stack, self.i_stack = self.stack_img(r_img, self.r_stack), self.stack_img(i_img, self.i_stack)
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

    def stack_img(self, img, stack):
        if stack is None:  # first step
            return np.tile(img[:, :, np.newaxis], (1,1,self.stack_num))
        else:  # next steps
            return np.concatenate((stack[:,:,1:], img[:, :, np.newaxis]), axis=2)

    def location_to_img(self, loc):
        coledges = range(-self.width//2, self.width//2+self.resolution, self.resolution)
        rowedges = range(0, self.height+self.resolution, self.resolution)
        col_vals, row_vals = loc[:, 0], loc[:, 1]
        H, _, _= np.histogram2d(x=row_vals, y=col_vals, bins=(rowedges, coledges))
        img = np.flip(H,axis=0)
        return img

    def get_state_space(self):
        return self.state_space.copy()