# ------- all imports ------
import configparser as prs
from os import read
import numpy as np
import random
from random import randint
from copy import deepcopy
from datetime import datetime


# -------- constants  --------

# the value indicates that a black cell
CELL_BLACK_ZERO = 0
CELL_BLACK_ONE = 1
CELL_BLACK_TOW = 2
CELL_BLACK_THREE = 3
CELL_BLACK_FOUR = 4
CELL_BLACK_FIVE = 5

# the value indicates that a cell can't put a bulb due to a zero black cell adjacent.
CELL_BULB_ZERO = 9
CELL_BULB_BAN = 9

# the value indicates that a white cell
CELL_EMPTY = 6

# the value indicates that a bulb in a cell
CELL_BULB = 7

# the value indicates that a cell is light up by a bulb
CELL_LIGHT = 8

class Board:
    black_info = []
    black_cells = []
    black_cells_unsatisfied = []
    black_cells_no_value = []
    banned_bulb = []
    placed_bulb = []

    def __init__(self, config_file, board_file):        

        # set up all parameters for EA and system, by reading config file
        self.ea_setup(config_file)

        # set up the board with black cells, by reading board file
        self.setup_board(board_file)

        set_random_seed(self.random_seed)

    def ea_setup(self, config_file):   

        self.config = prs.ConfigParser()
        self.config.sections()
        self.config.read(config_file)   
        
        # system default settings
        self.termination_evaluation = int(self.config["default"]["termination_evaluation"])
        self.termination_at_no_change = int(self.config["default"]["termination_at_no_change"])
        self.runs = int(self.config["default"]["runs_number"])
        self.random_seed = self.config["default"]["random_seed"]
        self.path_log = self.config["default"]["path_log"]
        self.path_solution = self.config["default"]["path_solution"]
        self.black_constraints = self.config["default"]["black_constraints"].lower() in ['true']

        # EA settings
        self.ea_mu = int(self.config["EA"]["mu"])
        self.ea_lambda = int(self.config["EA"]["lambda"])
        self.fitness_function = self.config["EA"]["fitness_function"]
        self.repair_function = self.config["EA"]["repair_function"].lower() in ['true']
        self.penalty_shrink_factor = float(self.config["EA"]["penalty_shrink_factor"])
        self.self_adaptive = self.config["EA"]["self_adaptive"].lower() in ['true']
        self.learning_rate = float(self.config["EA"]["learning_rate"])
        self.mutation_rate = float(self.config["EA"]["mutation_rate"])
        self.parent_selection = self.config["EA"]["parent_selection"]
        self.survival_selection = self.config["EA"]["survival_selection"]
        self.k_tournament_parent = int(self.config["EA"]["k_tournament_parents"])
        self.k_tournament_survival = int(self.config["EA"]["k_tournament_survivals"])


    # loading the board data of the problem
    # set up 3 variables:  
    #   - self.board_grids
    #   - self.banned_info
    #   - self.empty_cells 
    def setup_board(self, board_file):

        reading_file = open(board_file, 'r')    
        raw_data = reading_file.read().split("\n") 

        #  store first/second lines of data
        self.board_col = int(raw_data[0])
        self.board_row = int(raw_data[1])        

        # store data from the 3rd lines
        for i in range(2, len(raw_data)):
            message = raw_data[i].split(' ')
            try:
                test_int = [int(x) for x in message]
                self.black_info.append(test_int)
            except ValueError:
                pass

        # board_grids records very grid's information of the puzzle board
        self.board_grids = np.full((self.board_row,self.board_col), CELL_EMPTY)

        # create a list of the empty cells 
        self.empty_cells = []
        for i in range(self.board_row):
            for j in range(self.board_col):
                self.empty_cells.append([i, j])

        # update board_grids and empty_cells
        for x in self.black_info:
            row = x[1]-1
            col = x[0]-1
            self.board_grids[row][col] = x[2]
            self.empty_cells.remove([row, col])
            self.black_cells.append([row, col, x[2]])
            self.black_cells_no_value.append([row, col])
            

        

        reading_file.close()
        
    

    # update the boards by black number 0, the adjacents can't have bulbs
    # updated 3 variables:  
    #   - self.board_grids
    #   - self.banned_bulb
    #   - self.empty_cells    
    def set_black_zero(self):
        for x in self.black_cells:
            if x[2] == CELL_BLACK_ZERO:
                row = x[0]
                col = x[1]
                # self.empty_cells.remove([row, col])
                adjacents = self.get_adjacent_empty_cells(row, col)
                for y in adjacents:
                    self.empty_cells.remove(y)
                    self.banned_bulb.append(y)
                    self.board_grids[y[0]][y[1]] = CELL_BULB_BAN

        
        print("After adding ZERO Balck...")
        for i in range(len(self.board_grids)):
            print(self.board_grids[len(self.board_grids) - i - 1])


    # update the boards by black cells at the unique placement
    # updated 3 variables:  
    #   - self.board_grids
    #   - self.placed_bulb
    #   - self.empty_cells   
    #   - self.black_cells_unsatisfied
    def set_unique_bulbs(self):
        # remove Black 0 and 5
        blacks = deepcopy(self.black_cells)
        for x in blacks:
            if x[2] == CELL_BLACK_FIVE or x[2] == CELL_BLACK_ZERO:
                blacks.remove(x)

        # update the unique placment
        termination = False
        while not termination:
            termination = True
            for x in blacks:                
                empty_cell, bulb_cell = self.get_adjacent_black_and_empty(x[0], x[1])
                if len(empty_cell) + len(bulb_cell) == x[2]:
                    termination = False
                    blacks.remove(x)
                    for y in empty_cell:
                        self.place_unique_bulbs(y)

        # update the unsatisfied black cells   
        self.black_cells_unsatisfied = blacks
            
    

    # place the bulbs by unique placement
    def place_unique_bulbs(self, bulb):
        # for x in cells:
        if bulb in self.empty_cells:
            self.empty_cells.remove(bulb)
        self.placed_bulb.append(bulb)
        self.board_grids[bulb[0]][bulb[1]] = CELL_BULB
        self.setup_shining(bulb)
    

    # set shining cells by inserting a bulb
    # update:
    #   - self.board_grids 
    #   - self.empty_cells
    def setup_shining(self, pos):
        shining_cell = []

        # go right:
        for i in range(pos[1]+1, self.board_col):
            p = [pos[0], i]
            if p in self.black_cells_no_value:
                break
            shining_cell.append(p)

        # go up:
        for i in range(pos[0]+1, self.board_col):
            p = [i, pos[1]]
            if p in self.black_cells_no_value:
                break
            shining_cell.append(p)        

        # go right:
        i = pos[1] - 1
        while True:            
            p = [pos[0], i]
            if i < 0 or p in self.black_cells_no_value:
                break
            shining_cell.append(p)
            i -= 1

        # go down:
        i = pos[0] - 1
        while True:            
            p = [i, pos[1]]
            if i < 0 or p in self.black_cells_no_value:
                break
            shining_cell.append(p)
            i -= 1
        
        # update:  self.board_grids and self.empty_cells
        self.update_by_shining(shining_cell)


    #  shining cell update:  self.board_grids and self.empty_cells
    def update_by_shining(self, shinings):
        for c in shinings:
            if c in self.empty_cells:
                self.board_grids[c[0]][c[1]] = CELL_LIGHT            
                self.empty_cells.remove(c)
            elif c in self.banned_bulb:
                self.board_grids[c[0]][c[1]] = CELL_LIGHT  

        
    # get all empty cells
    def get_adjacent_black_and_empty(self, row, col) -> tuple:
        cells = get_adjacent_cells(row, col)
        adj_empty = [] 
        adj_bulb= []
        for x in cells:
            if x in self.empty_cells:
                adj_empty.append(x)
            if x in self.placed_bulb:
                adj_bulb.append(x)
        
        return adj_empty, adj_bulb
    

    # get all empty cells
    def get_adjacent_empty_cells(self, row, col) -> list:
        cells = get_adjacent_cells(row, col)
        adj = [] 
        for x in cells:
            if x in self.empty_cells:
                adj.append(x)        
        return adj

    
    # set empty cell probability 
    def empty_cell_prob(self):
        pass





# set random seed
def set_random_seed(seed):
    # set the random seed
    if seed == "time":
        random.seed(datetime.now())
    else:
        try:
            seed_int = int(seed)
        except ValueError:
            seed_int = 1
        random.seed(seed_int)



# get all adjacent cells on row and column, but they can be invalid cells
def get_adjacent_cells(row, col):
    return [[row, col+1], [row, col-1],
            [row+1, col], [row-1, col]] 
    
    
