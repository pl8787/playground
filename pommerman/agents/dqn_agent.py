"""
A Work-In-Progress agent using Tensorforce
"""
import json
from . import BaseAgent
from .. import characters

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class DQNAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, model_path=None, network_path=None, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.agent = None
        self.model_path = model_path
        self.network_path = network_path
        self.env = None
        self.trainable = True

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import DQNAgent as tf_DQNAgent

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

        with open(self.network_path, 'r') as fp:
            network = json.load(fp=fp)

        self.agent = tf_DQNAgent(
            states=dict(type='float', shape=[11, 11, 18]),
            actions=actions,
            discount=0.9,
            double_q_model=False,
            network=network,
            batching_capacity=1000,
            optimizer=dict(type='adam', learning_rate=1e-4),
            execution=dict(type='single', 
                           session_config=tf_config,
                           distributed_spec={})
                      )

        if self.model_path:
            print(self.model_path)
            self.agent.restore_model(directory="", file=self.model_path)

        return self.agent

    def act(self, obs, action_space):
        return self.agent.act(self.env.featurize_2d(obs))

