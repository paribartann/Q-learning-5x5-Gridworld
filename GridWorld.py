#<Environment for GridWorld (Example 3.8) present in R Sutton : Reinforcement Learning 
#using Gym AI.>
#Copyright (C) <October, 2018>  <Paribartan Dhakal>

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as
#published by the Free Software Foundation, either version 3 of the
#License, or (at your option) any later version.


#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.


import numpy as np
import sys
#from six import StringIO, b
from gym import utils
from gym.envs.toy_text import discrete



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#Action_prob = 0.25

"""
Here the gridworld is 5X5
where 'S' are the states where there is a reward of -1
If you are in 'A', whatever action you choose, it will take you to 'C' with a reward of +10
If you are in 'B', whatever action you choose, it will take you to 'D' with a reward of +5
gamma = 0.9
"""
MAPS = {
    "5x5": [
        "SASBF",
        "SSSSS",
        "SSSDS",
        "SSSSS",
        "SCSSS"
    ],
}

class GridWorldEnv(discrete.DiscreteEnv):


    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5x5", ):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 10)

        nA = 4
        nS = nrow * ncol


        #equal probability to start at S
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()



        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    
                    if letter in b'A':
                        rC, cC = 4, 1
                        sC = to_s(rC, cC)
                        li.append((1.0, sC, 10, False))
                        
                    elif letter in b'B':
                        rC, cC = 2, 3
                        sC = to_s(rC, cC)
                        li.append((1.0, sC, 5, False))
                        
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        if newstate==s:
                            rew = -1
                        else:
                            rew = 0
                            
                        #newletter = desc[newrow, newcol]
                        li.append((1.0, newstate, rew, False))

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile


