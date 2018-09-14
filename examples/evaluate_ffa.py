'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from tqdm import tqdm


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.FreezeAgent(),
        # agents.TFPPOAgent("/home/pangliang/nips/playground_pl/scripts/ppo_model/model"),
        # agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.RandomAgent(),
        agents.SimpleAgent(),
        # agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    # agent_list[0].initialize(env)

    rewards = []
    steps = []
    # Run the episodes just like OpenAI Gym
    for i_episode in tqdm(range(100)):
        state = env.reset()
        done = False
        step = 0
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            step += 1
            # print(actions)
        rewards.append(reward[0])
        steps.append(step)
        # print('Episode {} finished'.format(i_episode))
    env.close()
    print(np.mean(rewards))
    print(np.mean(steps))


if __name__ == '__main__':
    main()
