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
    banned_bulb = []
    placed_bulb = []

    def __init__(self, config_file, board_file):        

        # set up all parameters for EA and system, by reading config file
        self.ea_setup(config_file)

        # set up the board with black cells, by reading board file
        self.board_setup(board_file)

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
    def board_setup(self, board_file):

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
                self.empty_cells.remove([row, col])
                adjacents = self.get_adjacent_empty_cells(row, col)
                for y in adjacents:
                    self.empty_cells.remove(y)
                    self.banned_bulb.append(y)
                    self.board_grids[row][col] = CELL_BULB_BAN

    # update the boards by black cells at the unique placement
    # updated 3 variables:  
    #   - self.board_grids
    #   - self.placed_bulb
    #   - self.empty_cells   
    def set_unique_bulbs(self):

        # remove Black 0 and 5
        blacks = deepcopy(self.black_cells)
        for x in blacks:
            if x[2] == CELL_BLACK_FIVE or x[2] == CELL_BLACK_ZERO:
                blacks.remove(x)

        # update the unique placment
        termination = False
        while not termination:
            for x in blacks:                
                empty_cell, bulb_cell = self.get_adjacent_black_and_empty(x[0], x[1])
                if len(empty_cell) + len(bulb_cell) == x[2]:
                    blacks.remove(x)
                    for y in empty_cell:
                        self.place_bulbs(y)

    

    # place the bulbs by unique placement
    def place_bulbs(self, cells):
        for x in cells:
            self.empty_cells.remove(x)
            self.placed_bulb.append(x)
            self.board_grids[x[0]][x[1]] = CELL_BULB
            self.update_shining(x)
    

    def update_shining(self, position):
        pass





    # get all empty cells
    def get_adjacent_black_and_empty(self, row, col):
        cells = get_adjacent_cells(row, col)
        adj_empty = [] 
        adj_bulb= []
        for x in cells:
            if x in self.empty_cells:
                adj_empty.apend(x)
            if x in self.placed_bulb:
                adj_bulb.append(x)

        
        return adj_empty, adj_black
    

    # get all empty cells
    def get_adjacent_empty_cells(self, row, col):
        cells = get_adjacent_cells(row, col)
        adj = [] 
        for x in cells:
            if x in self.empty_cells:
                adj.apend(x)
        
        return adj




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
    
    

# initialize the map under validation
# all cells will fill up by bulbs if only unique way to do it
def initialize_validation_map(puzzle_map, rows, cols):
    # set Zero adjacent constraints
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle_map[i][j] == CELL_BLACK_ZERO:
                if i < rows - 1 and puzzle_map[i + 1][j] == CELL_EMPTY:
                    puzzle_map[i + 1][j] = CELL_BULB_ZERO
                if j < cols - 1 and puzzle_map[i][j + 1] == CELL_EMPTY:
                    puzzle_map[i][j + 1] = CELL_BULB_ZERO
                if i > 0 and puzzle_map[i - 1][j] == CELL_EMPTY:
                    puzzle_map[i - 1][j] = CELL_BULB_ZERO
                if j > 0 and puzzle_map[i][j - 1] == CELL_EMPTY:                
                        puzzle_map[i][j - 1] = CELL_BULB_ZERO

    # validating all unique bulbs
    new_bulb = True
    while new_bulb:
        new_bulb = False
        for i in range(0, rows):
            for j in range(0, cols):
                if 0 < puzzle_map[i][j] < CELL_BLACK_FIVE:
                    validating(puzzle_map, i, j, new_bulb)
                    

    return puzzle_map




def validating(pm, i, j, new_bulb):
    row = len(pm)
    col = len(pm[0])
    grids = []
    cell_empty = False
    if i < row - 1:
        if pm[i + 1][j] == CELL_EMPTY:
            cell_empty = True
            grids.append([i + 1, j])
        elif pm[i + 1][j] == CELL_BULB:
            grids.append([i + 1, j])
    if j < col - 1:
        if pm[i][j + 1] == CELL_EMPTY:
            cell_empty = True
            grids.append([i, j + 1])
        elif pm[i][j + 1] == CELL_BULB:
            grids.append([i, j + 1])
    if i > 0:
        if pm[i - 1][j] == CELL_EMPTY:
            cell_empty = True
            grids.append([i - 1, j])
        elif pm[i - 1][j] == CELL_BULB:
            grids.append([i - 1, j])
    if j > 0:
        if pm[i][j - 1] == CELL_EMPTY:
            cell_empty = True
            grids.append([i, j - 1])
        elif pm[i][j - 1] == CELL_BULB:
            grids.append([i, j - 1])
    if pm[i][j] == len(grids):
        for x in range(0, len(grids)):
            pm[grids[x][0]][grids[x][1]] = CELL_BULB


        # set cells lighted
        set_lighted = check_bulb_shining(pm, row, col)

        if set_lighted:
            print("Error in lightening white cells")
            # breakpoint()

        # ask for more loop in while since a new bulb has been set
        if cell_empty:
            new_bulb = True

    # check the empty cell which will be set as can't put a bulb in
    # because the number of black cell also reaches the requirement
    # only active when a new bulb has been set
    if new_bulb and 0 < pm[i][j] < CELL_BLACK_FOUR:
        cell_empty_adjacent = []
        cell_bulb_adjacent = 0
        if i < row - 1:
            if pm[i + 1][j] == CELL_EMPTY:
                cell_empty_adjacent.append([i + 1, j])
            elif pm[i + 1][j] == CELL_BULB:
                cell_bulb_adjacent += 1
        if j < col - 1:
            if pm[i][j + 1] == CELL_EMPTY:
                cell_empty_adjacent.append([i, j + 1])
            elif pm[i][j + 1] == CELL_BULB:
                cell_bulb_adjacent += 1
        if i > 0:
            if pm[i - 1][j] == CELL_EMPTY:
                cell_empty_adjacent.append([i - 1, j])
            elif pm[i - 1][j] == CELL_BULB:
                cell_bulb_adjacent += 1
        if j > 0:
            if pm[i][j - 1] == CELL_EMPTY:
                cell_empty_adjacent.append([i, j - 1])
            elif pm[i][j - 1] == CELL_BULB:
                cell_bulb_adjacent += 1

        if cell_bulb_adjacent and len(cell_empty_adjacent) and pm[i][j] == cell_bulb_adjacent:
            for x in range(0, len(cell_empty_adjacent)):
                row_update = cell_empty_adjacent[x][0]
                col_update = cell_empty_adjacent[x][1]
                pm[row_update][col_update] = CELL_BULB_ZERO


# check all white cells which has a bulb in it
# return - True: no two bulbs shine on each other
# return - False: at least two bulbs shine on each other
# @@ note @@ : puzzle_map will be updated by lighted info
def check_bulb_shining(puzzle_map, row, col):
    # validation = True
    # check the bulbs by every row
    conflict = 0
    for i in range(0, row):
        start = 0
        bulb = 0
        for j in range(0, col):
            cell = puzzle_map[i][j]
            if cell == CELL_BULB:
                bulb += 1
            if (cell < CELL_EMPTY) or (j == col - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[i][x] == CELL_EMPTY or puzzle_map[i][x] == CELL_BULB_ZERO:
                            puzzle_map[i][x] = CELL_LIGHT
                    bulb = 0
                start = j

    # check the bulbs by every column
    for i in range(0, col):
        start = 0
        bulb = 0
        for j in range(0, row):
            cell = puzzle_map[j][i]
            if cell == CELL_BULB:
                bulb += 1
            if (cell < CELL_EMPTY) or (j == row - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[x][i] == CELL_EMPTY or puzzle_map[x][i] == CELL_BULB_ZERO:
                            puzzle_map[x][i] = CELL_LIGHT
                    bulb = 0
                start = j

    return conflict