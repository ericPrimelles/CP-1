from email import policy
import gym
import numpy as np
from playGame import playGame

def valueIteration(env, gamma=0.99, epsilon=1e-04):
    
    V = np.zeros(env.observation_space.n) # Inicializando Valor de Estado como cero
    V_new = np.zeros(env.observation_space.n) # Copia del Valor de Estado
    policy = np.zeros(env.observation_space.n) # Inicializando política
    j = 0
    while True:
        
        diff = 0.0 # Inicializando diferencia límite
        
        for state in range(env.observation_space.n):
            v = V[state]
            action_values = np.zeros(env.action_space.n)
            for a  in range(env.action_space.n):
                suma = 0
                for i in env.P[state][a]:
                    prob, next_state, rwds, done = i
                    suma += prob* (gamma * V[next_state] + rwds)
                action_values[a] = suma
            V_new[state] = np.max(action_values)
            policy[state] = np.argmax(action_values)
            
            diff = max(diff, np.abs(v - V_new[state]))
        
        V = V_new.copy()
        j += 1
        print(f'Iteration {j}. Diff {diff}')
        if diff < epsilon:
            break;
    
    return V, policy
            
import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    
    env = gym.make("FrozenLake8x8-v1", is_slippery=True)
    env.reset()
    V, policy = valueIteration(env, epsilon=1e-07)
    print('\n')
    print(f'Value map: \n {V.reshape((8, 8))}')
    print(f'Policy: \n {policy.reshape(8,8)}')
    input('Press a key to continue...')
    
    sns.heatmap(V.reshape((8, 8)))
    plt.show()
    
    #playGame(env, 10, policy)
    
    playGame(env, 3, policy)
    
   
    
    
    
    