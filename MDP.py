import gym
from playGame import playGame
import numpy as  np

env = gym.make('FrozenLake8x8', is_slippery=False)
s = env.reset()

print(f'State Space: {env.observation_space}')
input("Press any key to continue....")
print(f'Action Space: {env.action_space}')
print("Actions: \n 0: Left \n 1: Down \n 2: Right \n 3: Up ")
input("Press any key to continue....")
print(f'Transition Model for state 0: \n Prob || Next state || Reward || Done? \n {env.P[0]}')
input("Press any key to continue....")

# Building a MDP
states = np.arange(0, env.observation_space.n) # States as numeration
actions = np.arange(0, env.action_space.n) # Actions as numeration
P = env.P # Transition Model includes rewards

s_0 = env.reset() # reset() method resets de environment and return the initial state
gamma = float(input("Input a Gamma Value ")) # Discount Factor
H = int(input('Input an Horizont value ')) # Horizont
input("Press any key to continue....")

MDP = {'States' : states, 'Actions': actions, 'P': P, "s_0": s_0, 'Gamma': gamma, 'H': H}
print (MDP)

input("Press any key to continue....")

playGame(env, 1, 'rnd')

env.close()

input("Press any key to continue....")

