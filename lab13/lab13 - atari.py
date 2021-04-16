import gym
# create environment
env = gym.make('SpaceInvaders-v0')

# reset environments
env.reset()

env.render()

is_done = False
env.reset()
while not is_done:
    action = env.action_space.sample()
    new_state, reward, is_done, info = env.step(action)
    print(info)
    env.render()