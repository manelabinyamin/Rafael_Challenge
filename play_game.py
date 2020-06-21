import numpy as np
# from Env_train import env_train
from Environments.Env import environment
import seaborn as sns
import matplotlib.pyplot as plt


def random_play(render=False, num_games=10):
    scores = []
    env = environment()
    for i in range(num_games):
        env.reset()
        done = False
        s = 0
        while not done:
            s+=1
            action = env.sample_action()
            observation, reward, done = env.step(action)
            if render:
                env.render()
        scores.append(env.total_score)
    return scores

scores = random_play(num_games=20)
# plt.figure()
# ax = plt.axes()
# ax.set_facecolor('w')
# sns.boxplot(x=scores, ax=ax)
# plt.show()