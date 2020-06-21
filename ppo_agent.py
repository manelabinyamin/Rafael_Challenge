class Agent():

    def __init__(self, env):
        self.env = env

    def run(self, policy, memory):
        state = self.env.reset()
        done = False
        while not done:
            # Running policy_old:
            action = policy.act(state, memory)
            state, reward, done, _ = self.env.step(action)

            # Saving is_terminal:
            # memory.rewards.append(reward)
            memory.is_terminals.append(done)

        # get ep rewards
        game_rewards = self.env.get_game_rewards()
        memory.rewards = game_rewards
        return memory
