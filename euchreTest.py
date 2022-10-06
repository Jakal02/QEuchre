"""
If you're interested in playing Euchre as a Human, 
this file is for you
"""

from rlcard.envs.euchre import EuchreEnv
from rlcard.models.euchre_rule_agent import EuchreRuleAgent
from rlcard.agents.human_agents.euchre_human_agent import EuchreHumanAgent

a = EuchreRuleAgent()
b = EuchreRuleAgent()
c = EuchreRuleAgent()
d = EuchreRuleAgent()

'''
This order determines the internal order of players but this
is not the same order that the game may be played in (starting clockwise from left of dealer)
as the dealer is randomly assigned at the start of the game. But when interpretting the payoffs,
remember that the numbers 0-3 correspond to the order you have placed them in the agents array
'''
agents = [a,b,c,d]
config = {
        'allow_step_back': False,
        'allow_raw_data': False,
        'single_agent_mode' : False,
        'active_player' : 0,
        'record_action' : False,
        'seed': None,
        'env_num': 1,
        }

test = EuchreEnv(config)
test.set_agents(agents)
trajectories,payoffs = test.run(is_training=False)
t = trajectories[0]

print(payoffs)