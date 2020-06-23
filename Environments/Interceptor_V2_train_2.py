# -*- coding: utf-8 -*-
"""
This is a training version of the game.
changes:
1. In this version the interceptors add the positive reward to the right step
2. In this version the desired distance to intercept the rockets starts very big and gradually decreases
"""

import numpy as np
import matplotlib.pyplot as plt


class World():
    width = 10000  # [m]
    height = 4000  # [m]
    dt = 0.2  # [sec]
    time = 0  # [sec]
    score = 0
    reward_city = -15
    reward_open = -1
    reward_fire = -1
    reward_intercept = 4
    g = 9.8  # Gravity [m/sec**2]
    fric = 5e-7  # Air friction [Units of Science]
    rocket_prob = 1  # expected rockets per sec
    def __init__(self):
        self.step_counter = 0
        self.rewards = []


class Turret():
    x = -2000  # [m]
    y = 0  # [m]
    x_hostile = 4800
    y_hostile = 0
    ang_vel = 30  # Turret angular speed [deg/sec]
    ang = 0  # Turret angle [deg]
    v0 = 800  # Initial speed [m/sec]
    min_prox_radius = 150
    reload_time = 1.5  # [sec]
    last_shot_time = -3  # [sec]

    def __init__(self):
        self.prox_radius = 150  # detonation proximity radius [m]

    def update(self, action_button):
        if action_button == 0:
            self.ang = self.ang - self.ang_vel * world.dt
            if self.ang < -90: self.ang = -90

        if action_button == 1:
            pass

        if action_button == 2:
            self.ang = self.ang + self.ang_vel * world.dt
            if self.ang > 90: self.ang = 90

        if action_button == 3:
            if world.time - self.last_shot_time > self.reload_time:
                Interceptor(world.step_counter)
                self.last_shot_time = world.time  # [sec]


class Interceptor():
    def __init__(self, step):
        global reward
        self.x = turret.x
        self.y = turret.y
        self.vx = turret.v0 * np.sin(np.deg2rad(turret.ang))
        self.vy = turret.v0 * np.cos(np.deg2rad(turret.ang))
        self.launch_step = step
        world.score = world.score + world.reward_fire
        world.rewards[world.step_counter] += world.reward_fire
        interceptor_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * world.fric * world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - world.g * world.dt
        self.x = self.x + self.vx * world.dt
        self.y = self.y + self.vy * world.dt
        if self.y < 0:
            Explosion(self.x, self.y)
            interceptor_list.remove(self)
        if np.abs(self.x) > world.width / 2:
            interceptor_list.remove(self)


class Rocket():
    def __init__(self, world):
        self.x = turret.x_hostile  # [m]
        self.y = turret.y_hostile  # [m]
        self.v0 = 700 + np.random.rand() * 300  # [m/sec]
        self.ang = -88 + np.random.rand() * 68  # [deg]
        self.vx = self.v0 * np.sin(np.deg2rad(self.ang))
        self.vy = self.v0 * np.cos(np.deg2rad(self.ang))
        rocket_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * world.fric * world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - world.g * world.dt
        self.x = self.x + self.vx * world.dt
        self.y = self.y + self.vy * world.dt


class City():
    def __init__(self, x1, x2, width):
        self.x = np.random.randint(x1, x2)  # [m]
        self.width = width  # [m]
        city_list.append(self)
        self.img = np.zeros((200, 800))
        for b in range(60):
            h = np.random.randint(30, 180)
            w = np.random.randint(30, 80)
            x = np.random.randint(1, 700)
            self.img[0:h, x:x + w] = np.random.rand()
        self.img = np.flipud(self.img)


class Explosion():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 500
        self.duration = 0.4  # [sec]
        self.verts1 = (np.random.rand(30, 2) - 0.5) * self.size
        self.verts2 = (np.random.rand(20, 2) - 0.5) * self.size / 2
        self.verts1[:, 0] = self.verts1[:, 0] + x
        self.verts1[:, 1] = self.verts1[:, 1] + y
        self.verts2[:, 0] = self.verts2[:, 0] + x
        self.verts2[:, 1] = self.verts2[:, 1] + y
        self.hit_time = world.time
        explosion_list.append(self)

    def update(self):
        if world.time - self.hit_time > self.duration:
            explosion_list.remove(self)


def Check_interception():
    for intr in interceptor_list:
        for r in rocket_list:
            if ((r.x - intr.x) ** 2 + (r.y - intr.y) ** 2) ** 0.5 < turret.prox_radius:
                rocket_list.remove(r)
                Explosion(intr.x, intr.y)
                if intr in interceptor_list: interceptor_list.remove(intr)
                world.score = world.score + world.reward_intercept
                world.rewards[intr.launch_step] += world.reward_intercept


def Check_ground_hit():
    for r in rocket_list:
        if r.y < 0:
            city_hit = False
            for c in city_list:
                if np.abs(r.x - c.x) < c.width:
                    city_hit = True
            if city_hit == True:
                world.score = world.score + world.reward_city
                world.rewards[world.step_counter] += world.reward_city
            else:
                world.score = world.score + world.reward_open
                world.rewards[world.step_counter] += world.reward_open
            Explosion(r.x, r.y)
            rocket_list.remove(r)


def Draw():
    plt.cla()
    plt.rcParams['axes.facecolor'] = 'black'
    for r in rocket_list:
        plt.plot(r.x, r.y, '.y')
    for intr in interceptor_list:
        plt.plot(intr.x, intr.y, 'or')
        C1 = plt.Circle((intr.x, intr.y), radius=turret.prox_radius, linestyle='--', color='gray', fill=False)
        ax = plt.gca()
        ax.add_artist(C1)
    for c in city_list:
        plt.imshow(c.img, extent=[c.x - c.width / 2, c.x + c.width / 2, 0, c.img.shape[0]])
        plt.set_cmap('bone')
    for e in explosion_list:
        P1 = plt.Polygon(e.verts1, True, color='yellow')
        P2 = plt.Polygon(e.verts2, True, color='red')
        ax = plt.gca()
        ax.add_artist(P1)
        ax.add_artist(P2)
    plt.plot(turret.x, turret.y, 'oc', markersize=12)
    plt.plot([turret.x, turret.x + 100 * np.sin(np.deg2rad(turret.ang))],
             [turret.y, turret.y + 100 * np.cos(np.deg2rad(turret.ang))], 'c', linewidth=3)
    plt.plot(turret.x_hostile, turret.y_hostile, 'or', markersize=12)
    plt.axes().set_aspect('equal')
    plt.axis([-world.width / 2, world.width / 2, 0, world.height])
    plt.title('Score: ' + str(world.score))
    plt.draw()
    plt.pause(0.001)


def Init():
    global world, turret, rocket_list, interceptor_list, city_list, explosion_list
    world = World()
    rocket_list = []
    interceptor_list = []
    turret = Turret()
    city_list = []
    explosion_list = []
    City(-world.width * 0.5 + 400, -world.width * 0.25 - 400, 800)  # (-4600, -2900)
    City(-world.width * 0.25 + 400, -400, 800)  # (-2100, -400)
    plt.rcParams['axes.facecolor'] = 'black'


def Game_step(action_button):
    world.rewards.append(0)
    world.time = world.time + world.dt

    if np.random.rand() < world.rocket_prob * world.dt:
        Rocket(world)

    for r in rocket_list:
        r.update()

    for intr in interceptor_list:
        intr.update()

    for e in explosion_list:
        e.update()

    turret.update(action_button)
    Check_interception()
    Check_ground_hit()

    r_locs = np.zeros(shape=(len(rocket_list), 4))
    for ind in range(len(rocket_list)):
        r_locs[ind, :] = [rocket_list[ind].x, rocket_list[ind].y, rocket_list[ind].vx ,rocket_list[ind].vy]

    i_locs = np.zeros(shape=(len(interceptor_list), 4))
    for ind in range(len(interceptor_list)):
        i_locs[ind, :] = [interceptor_list[ind].x, interceptor_list[ind].y, interceptor_list[ind].vx, interceptor_list[ind].vy]

    c_locs = np.zeros(shape=(len(city_list), 2))
    for ind in range(len(city_list)):
        c_locs[ind, :] = [city_list[ind].x, city_list[ind].width]

    world.step_counter += 1

    return r_locs, i_locs, c_locs, turret.ang, world.score


def Get_rewards():
    return world.rewards.copy()


def Get_score():
    return world.score