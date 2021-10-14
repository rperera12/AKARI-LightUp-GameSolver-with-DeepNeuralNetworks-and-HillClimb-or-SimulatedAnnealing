import setup as stp
from copy import deepcopy


class Placement:
    def __init__(self, board: stp.Board):
        self.board = deepcopy(board)
        self.bulbs_moveable = []
        self.empty_cells_initial = board.empty_cells
        
        
    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def compute_fitness(self):
        pass
    