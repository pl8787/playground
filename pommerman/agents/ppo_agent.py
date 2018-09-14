"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters


class TFPPOAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, model_path, character=characters.Bomber):
        super(TFPPOAgent, self).__init__(character)
        self.agent = None
        self.model_path = model_path
        self.env = None

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import PPOAgent

        self.env = env

        if type(env.action_space) == spaces.Tuple:
            actions = {
                    str(num): {
                    'type': int,
                    'num_actions': space.n
                }
                for num, space in enumerate(env.action_space.spaces)
            }
        else:
            actions = dict(type='int', num_actions=env.action_space.n)

        self.agent = PPOAgent(
            #states=dict(type='float', shape=env.observation_space.shape),
            states=dict(type='float', shape=[11, 11, 18]),
            actions=actions,
            #network=[
            #    dict(type='dense', size=64),
            #    dict(type='dense', size=64)
            #],
            network=[
                dict(type='conv2d', size=32),
                dict(type='conv2d', size=32),
                dict(type='conv2d', size=32),
                dict(type='conv2d', size=32),
                dict(type='flatten')
            ],
            batching_capacity=1000,
            step_optimizer=dict(type='adam', learning_rate=1e-4))
        print(self.model_path)
        self.agent.restore_model(directory="", file=self.model_path)

    def act(self, obs, action_space):
        return self.agent.act(self.env.featurize_2d(obs))

