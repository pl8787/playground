'''An agent that preforms a random action each step'''
from . import BaseAgent


class FreezeAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return 0
