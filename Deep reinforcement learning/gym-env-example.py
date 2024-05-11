import numpy as np
import matplotlib as plt

class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove(self.m*self.n-1)
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
        self.possibleActions = ['U','D','L','R']
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0
    
    def addMagicSqares(self, magicSquares):
        self.magicSquares = magicSquares
        i = 2
        for square in magicSquares:
            x = square // self.m
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            i += 1
            