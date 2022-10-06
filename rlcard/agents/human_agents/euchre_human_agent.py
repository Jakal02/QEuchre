from collections import defaultdict

import numpy as np

from rlcard.games.euchre.utils import LEFT, NON_TRUMP, ACTION_SPACE

class EuchreHumanAgent(object):
    
    def __init__(self,name="Default",mute_state = True):
        #Included for readability in debugger
        self.name = name

        self.mute_state = mute_state
        self.use_raw = False
    
    def step(self,state):
        # Right Now it gives you the option to select any action you want
        legal = state['raw_legal_actions']
        if not self.mute_state:
            print(state)
        
        # If you want to actually play as a human, uncomment the block comment
        else:
            if np.sum(state['flipped_choice']) == 0:
                print("Center Card:",state['flipped'])
            print("Your Hand:",state['hand'])
        

        # The state saves the actual object of the cards in state['center']... not very readable
        played=[]
        for card in state['center']:
            played.append(card.get_index())
        if len(played) != 0:
            print("Center Cards:",played)
        
        #print(f"I am player #{state['current_actor']}")
        print("Your Legal Actions:",legal)
        act = input("Select Legal Action")
        while act not in legal:
            act = input("Select Legal Action")
        return ACTION_SPACE[act]

    # Below was pasted from euchre_rule_agent

    # Used in env.py in run(). This is what calls the step function above
    # When you are not training.
    def eval_step(self, state):
        return self.step(state), []