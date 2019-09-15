import numpy as np


class CycleWorld:
    def __init__(self, num_states):
        self.cur_state = 0
        self.num_states = num_states
        
    def step(self, gammas=np.zeros(0)):
        self.cur_state = (self.cur_state + 1)%self.num_states
        action = 0

        num_predictions = 1 + len(gammas)

        y_tp1 = np.zeros(num_predictions)
        s_tp1 = np.zeros(1)

        if self.cur_state == self.num_states - 1:
            y_tp1[0] = 1
        if self.cur_state == 0:
            s_tp1[0] = 1
            
        for i in range(1, num_predictions):
            y_tp1[i] = pow(gammas[i-1], (self.num_states - 1) - self.cur_state)


        return self.createPhi(s_tp1, action), action, s_tp1, y_tp1
    
    def createPhi(self,state,action):
        phi = np.zeros(3)
        phi[0] = 1
        phi[1] = state[0]
        phi[2] = 1 - state[0]
        return phi
