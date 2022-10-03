from rlcard.envs import Env
from rlcard.games.euchre import Game
from rlcard.games.euchre.utils import ACTION_SPACE, ACTION_LIST
import numpy as np

class EuchreEnv(Env):

    def __init__(self, config):
        self.game = Game()
        self.name = "euchre"

        self.actions = ACTION_LIST
        self.state_shape = [len(self.actions)]
        super().__init__(config)
    

    def _extract_state(self, state):
        def vec(s):
            suit = {"C":[1,0,0,0], "D":[0,1,0,0], "H":[0,0,1,0], "S":[0,0,0,1]}
            rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
            if len(s)==1:
                return np.asarray(suit[s[0]] )
            else:
                return np.array( suit[s[0]] + [rank[s[1]]] )

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        '''
        structure of obs 
        suit = 4-1 Binary Feature | rank = 1 numerical Feature  
        1. Dealer pos relative to Agent:        4-1 Binary Feature
        2. suit of trump:                       4-1 Binary Feature
        3. Trump caller pos relative to Agent   4-1 Binary Feature
        4. Flipped Card                         4-1 Binary Feature and 1 numerical feature
        5. What happened to the flipped card    2-1 Binary Feature
            Avaliable = [0,0]
        6. The led suit for the hand            4-1 Binary Feature
        7. Center Cards                     4x  4-1 Binary Feature and 1 numerical feature
        7. Agents Hand                      6x  4-1 Binary Feature and 1 numerical feature
        8. Partners Hand                    5x  4-1 Binary Feature and 1 numerical feature
        9. Left Opponents Hand              5x  4-1 Binary Feature and 1 numerical feature
        10. Right Opponents Hand            5x  4-1 Binary Feature and 1 numerical feature
        '''

        obs = []
        curr_player_num = state['current_actor']
        # Save which player relative to you is the dealer
        '''1'''
        obs += [self._orderShuffler(curr_player_num,state['dealer_actor'])]

        '''2 and 3'''
        if state['trump'] is not None:
            obs += [ vec(state['trump']) ]
            obs += [self._orderShuffler(curr_player_num,state['calling_actor'])]
        else: # No Trump called
            obs += [ np.zeros(4) ]
            obs += [ np.zeros(4) ]
        
        '''4'''
        obs += [ vec(state['flipped']) ]
        '''5'''
        obs += [np.array(state['flipped_choice'])]
        '''6'''
        if state['lead_suit'] is not None:
            obs += [ vec(state['lead_suit']) ]
        else:
            obs += [ np.asarray([0,0,0,0]) ]
        
        '''7'''
        # TODO: Fix this. Done?
        obs += [ vec(e.get_index()) for e in state['center'] ]
        obs += [ np.zeros(5*(4-len(state['center'])))-1 ]

        '''8'''
        # TODO Fix this. Done?
        obs += [ vec(e) for e in state['hand'] ]
        obs += [ np.zeros(5*(6-len(state['hand'])))-1 ]

        '''
        Need to build 3 hands for each other player
        Note, their hands will grow as mine shrinks
        Because their 'hand' represents which cards I've seen them play
        '''
        '''9 10 11'''
        for i in range(1,4):
            rel_player_num = (i - curr_player_num + 4) % 4
            obs += [ vec(e) for e in state['played'][rel_player_num] ]
            obs += [ np.zeros(5*(5-len(state['center'])))-1 ]

        state['obs'] = np.hstack(obs)
        return state

    def _orderShuffler(self,curr_player_num, num):
            '''
            As a player, you see the game in this way:
                            Partner = 2
            Left opponent = 1       Right opponent = 3
                            You = 0
            The best players try to decuce which players have which cards as more are revealed.
            It is not so simple as to say "I've seen the King of Spades", you have to remember who threw it.
            Imagine my hand has the Ace of Spades and Ace of Hearts, there are two tricks left, and I lead, and lets say, diamonds are trump.
            I am trying to win the most tricks, which means I want to win THIS hand, if I can, while I lead.
            I want to guess which (if any) off-suits my opponents have. If my partner has thrown off spades before in a previous trick,
            there is a higher chance that they have no more, especially if it was the king of Spades. Thats a powerful signal and it
            tells me that if my partner has any off-suit it probably isn't spades, and if there are any left, the opponents probably have them.
            Playing an off-suit you know your partner doesn't have also gives them the flexibility to trump the first opponents card, or throw off garbage.
            Therefore, since I remembered my partner threw off the King of Spades I deduced they were out of Spades, and thus I threw the Ace of Spades, as
            spades have a higher chance of being in my opponents hands.
            Now consider the above if my Left opponent and Partner had switched hands. See how there is a difference?!

            Also, it's important to remember who was the dealer. As the dealer has an information advantage.
            '''
            bin_encode = np.zeros(4)
            adjusted_num = (num - curr_player_num + 4) % 4
            bin_encode[adjusted_num] = 1
            return bin_encode

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = [ACTION_SPACE[action] for action in legal_actions]
        return legal_ids

    def get_payoffs(self):
        return self.game.get_payoffs()