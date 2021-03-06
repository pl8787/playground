''' Helpers'''
import os
from .. import agents

USE_GAME_SERVERS = os.getenv("PLAYGROUND_USE_GAME_SERVERS")
GAME_SERVERS = {id_: os.getenv("PLAYGROUND_GAME_INSTANCE_%d" % id_)
                for id_ in range(4)}


# NOTE: This routine is meant for internal usage.
def make_agent_from_string(agent_string, agent_id, docker_env_dict=None):
    '''Internal helper for building an agent instance'''
    
    agent_type, agent_control = agent_string.split("::")

    assert agent_type in ["player", "random", "docker", "test", "tensorforce",
                          "train"]

    agent_instance = None

    if agent_type == "player":
        agent_instance = agents.PlayerAgent(agent_control=agent_control)
    elif agent_type == "random":
        agent_instance = agents.RandomAgent()
    elif agent_type == "docker":
        port = agent_id + 1000
        if not USE_GAME_SERVERS:
            server = 'http://localhost'
        else:
            server = GAME_SERVERS[agent_id]
        assert port is not None
        agent_instance = agents.DockerAgent(
            agent_control, port=port, server=server, env_vars=docker_env_dict)
    elif agent_type == "test":
        agent_instance = eval(agent_control)()
    elif agent_type == "tensorforce":
        agent_instance = agents.TensorForceAgent(algorithm=agent_control)
    elif agent_type == "train":
        agent_id, model_path, network_path = agent_control.split("+")
        print(agent_id, model_path, network_path)
        if model_path == '':
            model_path = None
        agent_instance = eval(agent_id)(model_path = model_path, 
                                        network_path = network_path)
        print(agent_instance)

    return agent_instance
