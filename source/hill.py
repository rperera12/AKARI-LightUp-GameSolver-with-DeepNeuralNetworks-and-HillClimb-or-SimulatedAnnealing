# import from project functions
import math
import operator

import initial as init

# import from outside functions
import random
from copy import deepcopy


# Map class stores all game properties related to the map
class Map:
    size = 0
    column = 0
    row = 0
    total_available_cells = 0
    fitness = 0
    board = []
    optimized_board = []
    black_cells = []
    available_cells = []
    bulb_running = []
    board_running = []

    result_log = []

    def __init__(self, map_file, config):
        self.board = []
        self.available_cells = []
        self.bulb_running = []
        self.board_running = []
        self.result_log = []
        self.size = 0
        self.row = 0
        self.column = 0
        self.fitness = 0

        self.file = map_file
        self.black_cells = self.read_map()
        self.original_board = self.load_board()
        self.board = self.original_board
        if config.black_constraints:
            self.optimized_board = self.validation_board()
        else:
            self.optimized_board = self.original_board
        self.board_running = deepcopy(self.optimized_board)
        self.available_cells = self.set_available_cells()
        self.total_available_cells = self.row * self.column - len(self.black_cells)

    # read the map data and store it
    def read_map(self):
        read_file = []  # original file data read
        puzzle = []  # initial puzzle will be returned

        with open(self.file) as fp:
            line = fp.readline()
            read_file.append(line)
            while line:
                line = fp.readline()
                read_file.append(line)

        #  store first/second lines of data
        self.column = int(read_file[0])
        self.row = int(read_file[1])

        length = len(read_file)

        # store data from the 3rd lines
        for i in range(2, length - 1):
            puzzle.append([])
            message = read_file[i].split(" ")
            for j in range(0, len(message)):
                puzzle[i - 2].append(int(message[j]))

        return puzzle

    def log_update(self, run, fitness):
        self.result_log.append([run, fitness])

    def create_board(self):
        board = []  # array will be returned
        for x in range(0, self.row):
            board.append([])
            for y in range(0, self.column):
                board[x].append(init.CELL_EMPTY)
        return board

    def load_board(self):

        new_board = self.create_board()
        numbers = len(self.black_cells)
        for i in range(0, numbers):
            col = self.black_cells[i][0] - 1
            row = self.black_cells[i][1] - 1
            value = self.black_cells[i][2]
            new_board[row][col] = value
        return new_board

    # set available cells for hill climbing
    def set_available_cells(self):
        cells = []
        for i in range(0, self.row):
            for j in range(0, self.column):
                if self.optimized_board[i][j] == init.CELL_EMPTY:
                    cells.append([i, j])
        return cells

    # this function will insert the bulbs to meet the requirement of black cell adjacency
    def set_start_map(self):
        cells_for_inserting = deepcopy(self.available_cells)
        for i in range(0, len(self.black_cells)):
            bulb_number = self.black_cells[i][2]
            if bulb_number > 0 and bulb_number != 5:
                neighbors = []

                # 1. upper -- row + 1
                upper = [self.black_cells[i][1], self.black_cells[i][0] - 1]
                if upper in cells_for_inserting:
                    neighbors.append(upper)
                    cells_for_inserting.remove(upper)

                # 2. down:-- row - 1
                down = [self.black_cells[i][1] - 2, self.black_cells[i][0] - 1]
                if down in cells_for_inserting:
                    neighbors.append(down)
                    cells_for_inserting.remove(down)

                # 3. down:-- column + 1
                right = [self.black_cells[i][1] - 1, self.black_cells[i][0]]
                if right in cells_for_inserting:
                    neighbors.append(right)
                    cells_for_inserting.remove(right)

                # 4. left:-- column - 1
                left = [self.black_cells[i][1] - 1, self.black_cells[i][0] - 2]
                if left in cells_for_inserting:
                    neighbors.append(left)
                    cells_for_inserting.remove(left)

                if neighbors:
                    if len(neighbors) < bulb_number:
                        bulb_number = len(neighbors)
                    bulb_inserting = random.sample(neighbors, k=bulb_number)

                    for x in range(0, bulb_number):
                        self.bulb_running.append(bulb_inserting[x])
                        self.board_running[bulb_inserting[x][0]][bulb_inserting[x][1]] = init.CELL_BULB

    # initialize the map under validation
    # all cells will fill up by bulbs if only unique way to do it
    def validation_board(self):
        # set Zero adjacent constraints
        new_board = deepcopy(self.board)
        rows = self.row
        cols = self.column
        for i in range(0, rows):
            for j in range(0, cols):
                if new_board[i][j] == init.CELL_BLACK_ZERO:
                    if i < rows - 1:
                        if new_board[i + 1][j] == init.CELL_EMPTY:
                            new_board[i + 1][j] = init.CELL_BULB_ZERO
                    if j < cols - 1:
                        if new_board[i][j + 1] == init.CELL_EMPTY:
                            new_board[i][j + 1] = init.CELL_BULB_ZERO
                    if i > 0:
                        if new_board[i - 1][j] == init.CELL_EMPTY:
                            new_board[i - 1][j] = init.CELL_BULB_ZERO
                    if j > 0:
                        if new_board[i][j - 1] == init.CELL_EMPTY:
                            new_board[i][j - 1] = init.CELL_BULB_ZERO

        # validating all unique bulbs
        new_bulb = True
        while new_bulb:
            new_bulb = False
            for i in range(0, rows):
                for j in range(0, cols):
                    if 0 < new_board[i][j] < init.CELL_BLACK_FIVE:
                        grids = []
                        cell_empty = False
                        if i < rows - 1:
                            if new_board[i + 1][j] == init.CELL_EMPTY:
                                cell_empty = True
                                grids.append([i + 1, j])
                            elif new_board[i + 1][j] == init.CELL_BULB:
                                grids.append([i + 1, j])
                        if j < cols - 1:
                            if new_board[i][j + 1] == init.CELL_EMPTY:
                                cell_empty = True
                                grids.append([i, j + 1])
                            elif new_board[i][j + 1] == init.CELL_BULB:
                                grids.append([i, j + 1])
                        if i > 0:
                            if new_board[i - 1][j] == init.CELL_EMPTY:
                                cell_empty = True
                                grids.append([i - 1, j])
                            elif new_board[i - 1][j] == init.CELL_BULB:
                                grids.append([i - 1, j])
                        if j > 0:
                            if new_board[i][j - 1] == init.CELL_EMPTY:
                                cell_empty = True
                                grids.append([i, j - 1])
                            elif new_board[i][j - 1] == init.CELL_BULB:
                                grids.append([i, j - 1])
                        if new_board[i][j] == len(grids):
                            for x in range(0, len(grids)):
                                new_board[grids[x][0]][grids[x][1]] = init.CELL_BULB

                            # set cells lighted
                            set_lighted = check_bulb_shining(new_board, rows, cols)

                            if set_lighted:
                                print("Error in lightening white cells")
                                # breakpoint()

                            # ask for more loop in while since a new bulb has been set
                            if cell_empty:
                                new_bulb = True

                        # check the empty cell which will be set as can't put a bulb in
                        # because the number of black cell also reaches the requirement
                        # only active when a new bulb has been set
                        if new_bulb:
                            if 0 < new_board[i][j] < init.CELL_BLACK_FOUR:
                                cell_empty_adjacent = []
                                cell_bulb_adjacent = 0
                                if i < rows - 1:
                                    if new_board[i + 1][j] == init.CELL_EMPTY:
                                        cell_empty_adjacent.append([i + 1, j])
                                    elif new_board[i + 1][j] == init.CELL_BULB:
                                        cell_bulb_adjacent += 1
                                if j < cols - 1:
                                    if new_board[i][j + 1] == init.CELL_EMPTY:
                                        cell_empty_adjacent.append([i, j + 1])
                                    elif new_board[i][j + 1] == init.CELL_BULB:
                                        cell_bulb_adjacent += 1
                                if i > 0:
                                    if new_board[i - 1][j] == init.CELL_EMPTY:
                                        cell_empty_adjacent.append([i - 1, j])
                                    elif new_board[i - 1][j] == init.CELL_BULB:
                                        cell_bulb_adjacent += 1
                                if j > 0:
                                    if new_board[i][j - 1] == init.CELL_EMPTY:
                                        cell_empty_adjacent.append([i, j - 1])
                                    elif new_board[i][j - 1] == init.CELL_BULB:
                                        cell_bulb_adjacent += 1

                                if cell_bulb_adjacent and len(cell_empty_adjacent):
                                    if new_board[i][j] == cell_bulb_adjacent:
                                        for x in range(0, len(cell_empty_adjacent)):
                                            row_update = cell_empty_adjacent[x][0]
                                            col_update = cell_empty_adjacent[x][1]
                                            new_board[row_update][col_update] = init.CELL_BULB_ZERO

        return new_board


# set bulbs into white cells
# return - array: map_data (every rows * cols)with bulbs in it
def set_random_bulbs(puzzle_map, white_cells, bulb_number):
    r_bulb_map = deepcopy(puzzle_map)

    # random select bulbs placement
    r_choice = random.sample(white_cells, k=bulb_number)

    # update map data with bulbs
    for i in range(0, bulb_number):
        r_row = r_choice[i][0]
        r_col = r_choice[i][1]
        r_bulb_map[r_row][r_col] = init.CELL_BULB

    return r_bulb_map


# count all numbers of :
# 1. black cells
# 2. shining cells
# 3. bulb cells
def evaluate_puzzle_map(board: Map):
    my_board = deepcopy(board.optimized_board)
    black_cells = deepcopy(board.black_cells)
    bulb_cells = deepcopy(board.bulb_running)

    number_cells_black = len(black_cells)
    number_cells_empty = 0
    number_cells_bulb = 0

    cols = board.column
    rows = board.row

    available_placement = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if my_board[i][j] == init.CELL_EMPTY:
                available_placement += 1

    puzzle = insert_bulbs(my_board, bulb_cells)

    fitness_shining_conflict = check_bulb_shining(puzzle, rows, cols)

    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle[i][j] == init.CELL_BULB:
                number_cells_bulb += 1
            if puzzle[i][j] == init.CELL_EMPTY:
                number_cells_empty += 1
            elif puzzle[i][j] == init.CELL_BULB_ZERO:
                number_cells_empty += 1

    number_cells_shining = rows * cols - number_cells_empty - number_cells_black
    fitness_black_conflict = check_black_bulb(puzzle, rows, cols, black_cells)

    # penalty function
    total_conflict = fitness_shining_conflict + fitness_black_conflict

    evaluation_fitness = int(100 * number_cells_shining / board.total_available_cells) - total_conflict * 3

    # print(f'shining bulbs:{number_cells_shining}  Total conflicts:{total_conflict}')
    # print(f'Evaluation fitness: {evaluation_fitness}')
    # print(f'board.total_available_cells: {board.total_available_cells}')
    # print(f'fitness_shining_conflict: {fitness_shining_conflict}')
    # print(f'fitness_black_conflict: {fitness_black_conflict}')

    return evaluation_fitness


# put the bulb array into puzzle map
def insert_bulbs(puzzle_map, bulbs):
    puzzle_copied = deepcopy(puzzle_map)
    for i in range(0, len(bulbs)):
        row = bulbs[i][0]
        col = bulbs[i][1]
        puzzle_copied[row][col] = init.CELL_BULB
    return puzzle_copied


# check black cell value by the number of surrounded bulbs
# return - Number of black cell number - bulbs conflicts
def check_black_bulb(map_data, number_rows, number_cols, black_data):
    conflict = 0
    black_cell_number = len(black_data)
    for i in range(1, black_cell_number):
        col = black_data[i][0] - 1
        row = black_data[i][1] - 1
        value = black_data[i][2]
        bulbs_surround = 0
        if col < number_cols - 1:
            if map_data[row][col + 1] == init.CELL_BULB:
                bulbs_surround += 1
        if col > 0:
            if map_data[row][col - 1] == init.CELL_BULB:
                bulbs_surround += 1
        if row < number_rows - 1:
            if map_data[row + 1][col] == init.CELL_BULB:
                bulbs_surround += 1
        if row > 0:
            if map_data[row - 1][col] == init.CELL_BULB:
                bulbs_surround += 1

        if bulbs_surround != value and value != init.CELL_BLACK_FIVE:
            conflict += abs(bulbs_surround - value)

    return conflict


# check the status of white cells whether light up or not
# return - True: all white cells are light up
# return - False: not all white cells are light up
def check_light_up(map_data, number_rows, number_cols):
    # count_white_cells = 0
    net_white_cells = []
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_EMPTY:
                net_white_cells.append([i, j])

    return net_white_cells


# check white cells which can put a bulb in
def check_net_cells(map_data, number_rows, number_cols):
    # count_white_cells = 0
    net_white_cells = []
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_EMPTY:
                net_white_cells.append([i, j])

    return net_white_cells


# read the map if bulbs are placed as valid
# return array: bulbs data with row/col info
#   0: number of bulbs
#   >0: col/row indicates bulb cell
def read_valid_map(map_data, number_rows, number_cols):
    puzzle_data = []
    count_bulb = 0  # count of bulbs
    count_lighted = 0  # count of lighted cell
    puzzle_data.append([0, 0])
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == init.CELL_BULB:
                puzzle_data.append([])
                puzzle_data[count_bulb + 1].append(j + 1)
                puzzle_data[count_bulb + 1].append(i + 1)
                count_bulb += 1
            elif map_data[i][j] == init.CELL_LIGHT:
                count_lighted += 1
    count_lighted += count_bulb
    puzzle_data[0][0] = count_bulb
    puzzle_data[0][1] = count_lighted
    return puzzle_data


# check all white cells which has a bulb in it
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
            if cell == init.CELL_BULB:
                bulb += 1
            if (cell < init.CELL_EMPTY) or (j == col - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[i][x] == init.CELL_EMPTY or puzzle_map[i][x] == init.CELL_BULB_ZERO:
                            puzzle_map[i][x] = init.CELL_LIGHT
                    bulb = 0
                start = j

    # check the bulbs by every column
    for i in range(0, col):
        start = 0
        bulb = 0
        for j in range(0, row):
            cell = puzzle_map[j][i]
            if cell == init.CELL_BULB:
                bulb += 1
            if (cell < init.CELL_EMPTY) or (j == row - 1):
                if bulb > 1:
                    conflict += bulb - 1
                if bulb:
                    for x in range(start, j + 1):
                        if puzzle_map[x][i] == init.CELL_EMPTY or puzzle_map[x][i] == init.CELL_BULB_ZERO:
                            puzzle_map[x][i] = init.CELL_LIGHT
                    bulb = 0
                start = j

    return conflict


# adding a bulb and generate 10 neighbors to choose a local optimal
def add_one_bulb(board: Map):
    possible_cells = deepcopy(board.available_cells)
    for i in range(0, len(board.bulb_running)):
        possible_cells.remove(board.bulb_running[i])
    boards = []

    for i in range(0, 10):
        my_board = deepcopy(board)
        if possible_cells:
            my_board.bulb_running.append(random.choice(possible_cells))
        my_board.fitness = evaluate_puzzle_map(my_board)
        boards.append(my_board)
    boards.sort(key=operator.attrgetter('fitness'), reverse=True)
    # print(f'Local optimal fitness by adding a bulb: {boards[0].fitness}')

    return boards[0]


# reduce a bulb and generate 10 neighbors to choose a local optimal
def reduce_one_bulb(board: Map):
    boards = []

    for i in range(0, 10):
        my_board = deepcopy(board)
        if my_board.bulb_running:
            my_board.bulb_running.remove(random.choice(my_board.bulb_running))
        my_board.fitness = evaluate_puzzle_map(my_board)
        boards.append(my_board)
    boards.sort(key=operator.attrgetter('fitness'), reverse=True)
    # print(f'Local optimal fitness by reducing a bulb: {boards[0].fitness}')

    return boards[0]


# moving a bulb to another empty cell, generate 10 neighbors to choose a local optimal
def moving_one_bulb(board: Map):
    possible_cells = deepcopy(board.available_cells)
    for i in range(0, len(board.bulb_running)):
        possible_cells.remove(board.bulb_running[i])
    boards = []

    for i in range(0, 10):
        my_board = deepcopy(board)
        if my_board.bulb_running:
            my_board.bulb_running.remove(random.choice(my_board.bulb_running))
        if possible_cells:
            my_board.bulb_running.append(random.choice(possible_cells))
        my_board.fitness = evaluate_puzzle_map(my_board)
        boards.append(my_board)
    boards.sort(key=operator.attrgetter('fitness'), reverse=True)
    # print(f'Local optimal fitness by moving a bulb: {boards[0].fitness}')

    return boards[0]


# hill climb: choose best one of adding/removing/moving a bulb
def hill_climb(board: Map, configuration):
    if configuration.moving_action:
        results = [add_one_bulb(board), reduce_one_bulb(board), moving_one_bulb(board)]
    else:
        results = [add_one_bulb(board), reduce_one_bulb(board)]

    results.sort(key=operator.attrgetter('fitness'), reverse=True)
    # print(f'The current local optimal: {results[0].fitness}')

    if results[0].fitness <= board.fitness:
        if not configuration.annealing:
            results[0] = board
        else:
            results[0] = anneal(board, results[0], configuration.annealing_temp)

    return results[0]


# annealing algorithm
def anneal(f1, f2, temp):
    p = random.uniform(0, 1)
    d = -abs(f1.fitness - f2.fitness) / temp
    annealing = math.exp(d)
    if p > annealing:
        return f2
    return f1

