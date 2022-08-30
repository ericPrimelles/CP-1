import numpy as np

def policyIteration(env, gamma=0.99, epsilon=1e-07):
    V = np.zeros(env.observation_space.n)
    V_new = V.copy()
    policy = np.random.randint(0, env.action_space.n, env.observation_space.n)
    policy_old = np.zeros(env.observation_space.n)
    
    k = 0
    
    while True:
        # Policy evaluation
        while True:
            diff = 0.0
            for s in range(env.observation_space.n):
                v = V[s]
                suma = 0.0
                
                for i in env.P[s][int(policy[s])]:
                    prob, state, rwd, _ = i
                    suma += prob * (gamma * V[state] + rwd)
                
                V_new[s] = suma
                diff = np.max([diff, v - V_new[s]])
            V = V_new.copy()
            if diff < epsilon:
                break
        # Policy improvment
        policy_old = policy.copy()
        
        for s in range(env.observation_space.n):
            action_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                suma = 0.0
                for i in env.P[s][action]:
                    prob, state, rwd,_ = i
                    suma += prob * (gamma * V[state]  + rwd)
                
                action_values[action] = suma
            
            policy[s] = np.argmax(action_values)
        k += 1
        print(f'Iteration: {k} Mean difference: {np.abs(np.mean(policy - policy_old))}')
        if np.array_equal(policy, policy_old):
            break
    
    
    return V, policy

import gym
import matplotlib.pyplot as plt
import seaborn as sns
from playGame import playGame
if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)
    env.reset()
    V, policy = policyIteration(env)
    print('\n')
    print(f'Value map: \n {V.reshape((8, 8))}')
    print(f'Policy: \n {policy.reshape(8,8)}')
    input('Press a key to continue...')
    
    sns.heatmap(V.reshape((8, 8)))
    plt.show()
    
    #playGame(env, 10, policy)
    
    playGame(env, 3, policy)
