import gym
import math


env = gym.make('CartPole-v1')

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finshed after {} timesteps".format(t+1))
            break
env.close()



# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
#     print(env.step.__dict__)
# env.close()

# from gym import spaces
# space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
# a = space.sample()
# assert space.contains(x)
# assert space.n == 8
