def playGame(env, episodes, policy):
    wins = 0
    if policy == 'rnd':
        for i in range(episodes):
            s = env.reset()
            env.render()
            done = False
            
            while not done:
                s, rwrd, done, _ = env.step(env.action_space.sample())
                env.render()
            
            if rwrd > 0:
                wins += 1
                print(f"Episode {i}: Win!!!!")
            else:
                print(f"Episode {i}: Lose!!!!")
            return    
        
   
    for i in range(episodes):
        s = env.reset()
        env.render()
        done = False
        
        while not done:
            s, rwrd, done, _ = env.step(int(policy[s]))
            env.render()
        
        if rwrd > 0:
            wins += 1
            print(f"Episode {i}: Win!!!!")
        else:
            print(f"Episode {i}: Lose!!!!")
            
    print(f"Victories: {wins}")