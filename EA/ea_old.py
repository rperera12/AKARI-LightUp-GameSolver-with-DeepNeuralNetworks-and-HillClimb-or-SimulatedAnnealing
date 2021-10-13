# the value indicates that a black cell
CELL_BLACK_ZERO = 0
CELL_BLACK_ONE = 1
CELL_BLACK_TOW = 2
CELL_BLACK_THREE = 3
CELL_BLACK_FOUR = 4
CELL_BLACK_FIVE = 5

# the value indicates that a cell can't put a bulb due to a zero black cell adjacent.
CELL_BULB_ZERO = 9

# the value indicates that a white cell
CELL_EMPTY = 6

# the value indicates that a bulb in a cell
CELL_BULB = 7

# the value indicates that a cell is light up by a bulb
CELL_LIGHT = 8


# -------- global variables --------


# ======== ============= ========
# ======== all functions ========
# ======== ============= ========



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
        r_bulb_map[r_row][r_col] = CELL_BULB

    return r_bulb_map


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


# count all numbers of :
# 1. black cells
# 2. shining cells
# 3. bulb cells
def evaluate_puzzle_map(puzzle_map, black_cells, bulb_cells, config):
    number_cells_black = len(black_cells) - 1
    number_cells_empty = 0
    number_cells_bulb = 0

    cols = black_cells[0][0]
    rows = black_cells[0][1]

    available_placement = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle_map[i][j] == CELL_EMPTY:
                available_placement += 1

    puzzle = insert_bulbs(puzzle_map, bulb_cells)

    fitness_shining_conflict = check_bulb_shining(puzzle, rows, cols)
    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle[i][j] == CELL_BULB:
                number_cells_bulb += 1
            if puzzle[i][j] == CELL_EMPTY:
                number_cells_empty += 1
            elif puzzle[i][j] == CELL_BULB_ZERO:
                number_cells_empty += 1

    number_cells_shining = rows * cols - number_cells_empty - number_cells_black
    fitness_black_conflict = check_black_bulb(puzzle, rows, cols, black_cells)

    # penalty function
    total_conflict = fitness_shining_conflict + fitness_black_conflict
    total_available_cell = rows * cols - number_cells_black
    shrink = config["penalty_shrink_factor"]
    minus = config["penalty_minus_factor"]
    if total_conflict > available_placement:
        total_conflict = available_placement - 1
    minor_factor = (available_placement - total_conflict * minus) / available_placement
    if minor_factor < 0:
        minor_factor = 0
    evaluation_fitness = int(100 * number_cells_shining * minor_factor / total_available_cell)
    original_fitness = int(100 * number_cells_shining / total_available_cell)
    if total_conflict:
        if config["fitness_function"] == "original":
            original_fitness = evaluation_fitness
            evaluation_fitness = 0
        else:
            evaluation_fitness = int(evaluation_fitness * shrink)

    puzzle_eval_data = {
        "black_cells": number_cells_black,
        "white_cells": rows * cols - number_cells_black,
        "empty_cells": number_cells_empty,
        "total_conflict": total_conflict,
        "original_fitness": original_fitness,
        "evaluation_fitness": evaluation_fitness,
        "number_cells_shining": number_cells_shining,
        "bulb_cells": len(bulb_cells),
        "bulb_cells_total": number_cells_bulb,
        "bulbs": bulb_cells
    }
    #    print(puzzle_eval_data)
    #    breakpoint()

    return puzzle_eval_data


# put the bulb array into puzzle map
def insert_bulbs(puzzle_map, bulbs):
    puzzle_copied = deepcopy(puzzle_map)
    for i in range(0, len(bulbs)):
        row = bulbs[i][0]
        col = bulbs[i][1]
        puzzle_copied[row][col] = CELL_BULB
    return puzzle_copied


# check black cell value by the number of surrounded bulbs
# return - True: bulbs adjacent fit
# return - False: bulbs adjacent don't fit
def check_black_bulb(map_data, number_rows, number_cols, black_data):
    conflict = 0
    black_cell_number = len(black_data)
    for i in range(1, black_cell_number):
        col = black_data[i][0] - 1
        row = black_data[i][1] - 1
        value = black_data[i][2]
        bulbs_surround = 0
        if col < number_cols - 1:
            if map_data[row][col + 1] == CELL_BULB:
                bulbs_surround += 1
        if col > 0:
            if map_data[row][col - 1] == CELL_BULB:
                bulbs_surround += 1
        if row < number_rows - 1:
            if map_data[row + 1][col] == CELL_BULB:
                bulbs_surround += 1
        if row > 0:
            if map_data[row - 1][col] == CELL_BULB:
                bulbs_surround += 1

        if bulbs_surround != value and value != CELL_BLACK_FIVE:
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
            if map_data[i][j] == CELL_EMPTY:
                net_white_cells.append([i, j])

    return net_white_cells


# check white cells which can put a bulb in
def check_net_cells(map_data, number_rows, number_cols):
    # count_white_cells = 0
    net_white_cells = []
    for i in range(0, number_rows):
        for j in range(0, number_cols):
            if map_data[i][j] == CELL_EMPTY:
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
            if map_data[i][j] == CELL_BULB:
                puzzle_data.append([])
                puzzle_data[count_bulb + 1].append(j + 1)
                puzzle_data[count_bulb + 1].append(i + 1)
                count_bulb += 1
            elif map_data[i][j] == CELL_LIGHT:
                count_lighted += 1
    count_lighted += count_bulb
    puzzle_data[0][0] = count_bulb
    puzzle_data[0][1] = count_lighted
    return puzzle_data


def log_text_add(log_data, log_file_path):
    with open(log_file_path, 'a+') as wf:
        string_to_write = list()
        string_to_write.append(f'\n[Running date and time]\n"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"\n')
        string_to_write.append(f'\n[Config Information]')
        for pair in log_data.items():
            string_to_write.append(f'\n  {pair}"')
        string_to_write.append(f'\n[Running results]\n')
        wf.writelines(string_to_write)
    wf.close()


# create population pool
def create_population_pool(net_white_cell, size):
    r_puzzle_data = []
    total_number_bulb = len(net_white_cell)

    loop = 0
    loop_warning = 0

    while loop < size:
        number_bulb = randint(1, total_number_bulb)
        new_bulb_set = random.sample(net_white_cell, k=number_bulb)
        new_bulb_set.sort()
        loop_warning += 1

        if new_bulb_set not in r_puzzle_data:
            r_puzzle_data.append(new_bulb_set)
            loop += 1

        if loop_warning > 100000:
            print("Creating population pool failure...")
            print("reduce the number of population size: mu")
            loop_warning = 0

    return r_puzzle_data


# recombine both parents to generate an offspring
# single crossover
def recombination(gene_space, parent_one, parent_two, cross_point_percent):
    cross_point = int(len(gene_space) * cross_point_percent)
    # print(f"cross point: {cross_point}")
    offspring = []
    for i in range(0, len(parent_one)):
        if parent_one[i] in gene_space[0:cross_point]:
            offspring.append(parent_one[i])
    for i in range(0, len(parent_two)):
        if parent_two[i] in gene_space[cross_point:]:
            offspring.append(parent_two[i])

    return offspring


# mutate an offspring
def mutation(gene_space, offspring, number_flip):
    mutation_offspring = deepcopy(offspring)

    mutation_gene = random.sample(gene_space, k=number_flip)

    for i in range(0, len(mutation_gene)):
        if mutation_gene[i] in offspring:
            mutation_offspring.remove(mutation_gene[i])
        else:
            mutation_offspring.append(mutation_gene[i])

    mutation_offspring.sort()

    # print(f"mutation list: {mutation_gene}")
    # print(f"before:{offspring}  after: {mutation_offspring}")

    return mutation_offspring


# select parents by fitness proportional
def roulette_wheel_selection(population_pool, number_pair_parents):
    pool = deepcopy(population_pool)
    parents_pool = []
    sum_fitness_mu = 0

    for i in range(0, len(pool)):
        sum_fitness_mu += pool[i].get("evaluation_fitness")

    for x in range(0, number_pair_parents):
        parents = []
        pool_run = deepcopy(pool)
        sum_fitness = sum_fitness_mu
        for i in range(0, 2):
            sum_temporary = 0
            parent_selection_value = randint(0, sum_fitness)
            for j in range(0, len(pool_run)):
                sum_temporary += pool_run[j].get("evaluation_fitness")
                if sum_temporary >= parent_selection_value:
                    sum_fitness = sum_fitness - pool_run[j].get("evaluation_fitness")
                    parents.append(pool_run[j])
                    del pool_run[j]
                    break
        parents_pool.append(parents)

    return parents_pool


# select parents by fitness proportional
def stochastic_universal_sampling(population_pool, number_pair_parents):
    pool = deepcopy(population_pool)
    parents_pool = []
    sum_fitness_mu = 0

    for i in range(0, len(pool)):
        sum_fitness_mu += pool[i].get("evaluation_fitness")
    sampling_r = random.uniform(0, float(sum_fitness_mu / number_pair_parents))

    sum_temporary = pool[0].get("evaluation_fitness")
    i = 0
    for _ in range(0, number_pair_parents):
        parents = []
        for _ in range(0, 2):
            sum_fitness_one_round = 0
            while sum_temporary < sampling_r:
                i += 1
                if i >= len(population_pool):
                    i = 0
                sum_temporary += pool[i].get("evaluation_fitness")
                sum_fitness_one_round += pool[i].get("evaluation_fitness")
            parents.append(pool[i])
            sampling_r += sum_fitness_one_round
            # print(sum_fitness_one_round)
            # print(sampling_r)

            if sampling_r >= sum_fitness_mu:
                i = 0
                sampling_r = random.uniform(0, float(sum_fitness_mu / number_pair_parents))
                sum_temporary = pool[0].get("evaluation_fitness")

        parents_pool.append(parents)

    return parents_pool


# uniform random
def uniform_random_parents_selection(population_pool, number_pair_selection, replacement):
    number_selection = number_pair_selection * 2
    if not replacement:
        if number_selection <= len(population_pool):
            pool = random.sample(population_pool, k=number_selection)
        else:
            pool = random.choices(population_pool, k=number_selection)
    else:
        pool = random.choices(population_pool, k=number_selection)

    parent_pool = []
    for i in range(0, number_pair_selection):
        parents = [pool[2 * i], pool[2 * i + 1]]
        parent_pool.append(parents)

    return parent_pool


# uniform random survival selection
def uniform_random_survival_selection(population_pool, number_selection, replacement):
    if not replacement:
        if number_selection <= len(population_pool):
            pool = random.sample(population_pool, k=number_selection)
        else:
            pool = random.choices(population_pool, k=number_selection)
    else:
        pool = random.choices(population_pool, k=number_selection)

    return pool


# k-Tournament Selection with or without replacement
def k_tournament_selection(population_pool, k, number_selection, replacement):
    pool = deepcopy(population_pool)
    survival = []
    for i in range(0, number_selection):
        survival_selection_pool = random.sample(pool, k=k)
        survival_selection_pool = sorted(survival_selection_pool, key=itemgetter('evaluation_fitness'), reverse=True)
        survival.append(survival_selection_pool[0])
        if not replacement:
            pool.remove(survival_selection_pool[0])
    return survival


# parent selections
def parent_selection(popular_pool, number_pair_parents, parent_selection_mode, k_tournament_parents):
    if parent_selection_mode == "roulette_wheel":
        parents_pool = roulette_wheel_selection(popular_pool,
                                                number_pair_parents)

    elif parent_selection_mode == "stochastic_universal_sampling":
        parents_pool = stochastic_universal_sampling(popular_pool,
                                                     number_pair_parents)

    elif parent_selection_mode == "uniform_random":
        parents_pool = uniform_random_parents_selection(popular_pool,
                                                        number_pair_parents,
                                                        replacement=True)

    else:  # elif parent_selection == "k_tournament":
        parents_pool = []
        for i in range(0, number_pair_parents):
            parents_pool.append(k_tournament_selection(popular_pool,
                                                       k=k_tournament_parents,
                                                       number_selection=2,
                                                       replacement=False))

    return parents_pool


def mutation_self_adaptive(config, mutation_rate, lambda_size, net_white_cells):
    t = 1 / math.sqrt(len(net_white_cells))

    if len(mutation_rate) < lambda_size:
        mean = float(config["initial_mutation_rate"])
        standard_deviation = 0.1

    else:
        standard_deviation = statistics.stdev(mutation_rate)
        mean = statistics.mean(mutation_rate)

    random_factor = float(numpy.random.random(1)[0])
    standard_deviation_new = standard_deviation * math.exp(random_factor * t)

    new_mutation_rate = abs(float(numpy.random.normal(mean, standard_deviation_new, 1)))

    return new_mutation_rate


# generate offspring, under recombination and mutation
def generate_offspring(net_white_cells, lambda_size, parents_pool, puzzle_map, puzzle_initial, config, mutation_rate):
    population_offspring = []
    for i in range(0, lambda_size):
        offspring = recombination(net_white_cells,
                                  parents_pool[i][0].get("bulbs"),
                                  parents_pool[i][1].get("bulbs"),
                                  cross_point_percent=0.5)

        if config["self_adaptive"]:
            evaluation_1 = evaluate_puzzle_map(puzzle_map,
                                               puzzle_initial,
                                               offspring, config=config)

            current_mutation_rate = mutation_self_adaptive(config, mutation_rate, lambda_size, net_white_cells)

            # print(f"mutation rate is {current_mutation_rate}")
            offspring = mutation(net_white_cells, offspring, int(current_mutation_rate * len(net_white_cells)))
            # evaluate offsprings
            evaluation_2 = evaluate_puzzle_map(puzzle_map,
                                               puzzle_initial,
                                               offspring, config=config)

            if evaluation_1.get("evaluation_fitness") < evaluation_2.get("evaluation_fitness"):
                mutation_rate.append(current_mutation_rate)

        else:
            offspring = mutation(net_white_cells, offspring,
                                 int(config["initial_mutation_rate"] * len(net_white_cells)))
            evaluation_2 = evaluate_puzzle_map(puzzle_map,
                                               puzzle_initial,
                                               offspring, config=config)
        if config["repair_function"]:
            if evaluation_2.get("total_conflict"):
                puzzle = insert_bulbs(puzzle_map, offspring)

                # print(f' before repair')
                # print(offspring)
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])
                invalid_bulbs = repair_bulb_shining(puzzle, puzzle_initial)
                for x in range(0, len(invalid_bulbs)):
                    if invalid_bulbs[x] in offspring:
                        offspring.remove(invalid_bulbs[x])
                # print(offspring)
                puzzle = insert_bulbs(puzzle_map, offspring)
                # print(f' repair shining')
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])
                bulbs = repair_black_bulb(puzzle, puzzle_initial)
                # puzzle = insert_bulbs(puzzle_map, bulbs)
                # print(f' repair black')
                # for x in range(0, len(puzzle)):
                #     print(puzzle[len(puzzle) - x - 1])

                evaluation_2 = evaluate_puzzle_map(puzzle_map,
                                                   puzzle_initial,
                                                   bulbs, config=config)

        population_offspring.append(evaluation_2)

    return population_offspring


# survival selections
def survival_selection(config, population_mu, population_lambda):
    mu_size = len(population_mu)

    if config["survival_strategy"] == "plus":
        survival_pool = population_mu + population_lambda
    else:  # comma
        survival_pool = population_lambda

    if config["survival_selection"] == "truncation":
        population_mu = truncation(survival_pool, number_survival=mu_size)

    elif config["survival_selection"] == "uniform_random":
        population_mu = uniform_random_survival_selection(survival_pool,
                                                          mu_size,
                                                          replacement=False)

    else:  # puzzle_config["survival_selection"] == "truncation":
        population_mu = k_tournament_selection(survival_pool,
                                               k=config["k_tournament_survivals"],
                                               number_selection=mu_size,
                                               replacement=False)

    return population_mu


# fix bulbs shining each other
# return - the valid bulbs
def repair_bulb_shining(puzzle_map, initial_puzzle):
    # validation = True
    # check the bulbs by every row
    puzzle = deepcopy(puzzle_map)
    row = initial_puzzle[0][0]
    col = initial_puzzle[0][1]
    invalid_bulbs = []
    for i in range(0, row):
        bulb = 0
        bulb_in_row = []
        for j in range(0, col):
            cell = puzzle[i][j]
            if cell == CELL_BULB:
                bulb += 1
                bulb_in_row.append([i, j])
            if (cell < CELL_EMPTY) or (j == col - 1):
                if bulb > 1:
                    delete_bulbs = random.sample(bulb_in_row, k=bulb - 1)
                    for x in range(0, len(delete_bulbs)):
                        invalid_bulbs.append(delete_bulbs[x])
                bulb = 0
                bulb_in_row = []

    # check the bulbs by every column
    for i in range(0, col):
        bulb = 0
        bulb_in_column = []
        for j in range(0, row):
            cell = puzzle[j][i]
            if cell == CELL_BULB:
                bulb += 1
                bulb_in_column.append([j, i])
            if (cell < CELL_EMPTY) or (j == row - 1):
                if bulb > 1:
                    delete_bulbs = random.sample(bulb_in_column, k=bulb - 1)
                    for x in range(0, len(delete_bulbs)):
                        if delete_bulbs[x] not in invalid_bulbs:
                            invalid_bulbs.append(delete_bulbs[x])
                bulb_in_column = []
                bulb = 0

    return invalid_bulbs


# check black cell value by the number of surrounded bulbs
def repair_black_bulb(puzzle_map, initial_puzzle):
    bulbs_surrounding = []
    number_rows = initial_puzzle[0][0]
    number_cols = initial_puzzle[0][1]
    black_cell_number = len(initial_puzzle)
    bulbs = []
    for i in range(1, black_cell_number):
        col = initial_puzzle[i][0] - 1
        row = initial_puzzle[i][1] - 1
        value = initial_puzzle[i][2]
        bulbs_surround = 0
        empty_cells = []
        bulb_cells = []
        if col < number_cols - 1:
            if puzzle_map[row][col + 1] == CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row, col + 1])
            elif puzzle_map[row][col + 1] == CELL_EMPTY:
                empty_cells.append([row, col + 1])
        if col > 0:
            if puzzle_map[row][col - 1] == CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row, col - 1])
            elif puzzle_map[row][col - 1] == CELL_EMPTY:
                empty_cells.append([row, col - 1])
        if row < number_rows - 1:
            if puzzle_map[row + 1][col] == CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row + 1, col])
            elif puzzle_map[row + 1][col] == CELL_EMPTY:
                empty_cells.append([row + 1, col])
        if row > 0:
            if puzzle_map[row - 1][col] == CELL_BULB:
                bulbs_surround += 1
                bulb_cells.append([row - 1, col])
            elif puzzle_map[row - 1][col] == CELL_EMPTY:
                empty_cells.append([row - 1, col])

        if value != CELL_BLACK_FIVE:

            if bulbs_surround > value:
                bulbs_surrounding = random.sample(bulb_cells, k=value)
            elif bulbs_surround < value:
                bulbs_surrounding = random.sample(empty_cells, k=value - bulbs_surround) + bulb_cells
            else:
                bulbs_surrounding = bulb_cells
        else:
            bulbs_surrounding = bulb_cells

        for x in range(0, len(bulbs_surrounding)):
            bulbs.append(bulbs_surrounding[x])

    return bulbs


# write the solution to file
def write_solution(initial_puzzle, solution, file_path):
    rows = initial_puzzle[0][0]
    cols = initial_puzzle[0][1]

    bulb = []
    lighted = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if solution[i][j] == CELL_BULB:
                bulb.append([j + 1, i + 1])
                lighted += 1
            if solution[i][j] == CELL_LIGHT:
                lighted += 1
    bulb.sort()
    with open(file_path, 'a+') as wf:
        wf.write(f'{initial_puzzle[0][0]}\n'
                 f'{initial_puzzle[0][1]}\n')
        for i in range(1, len(initial_puzzle)):
            wf.write(f'{initial_puzzle[i][0]} {initial_puzzle[i][1]} {initial_puzzle[i][2]}\n')
        wf.write(f'{str(lighted)}\n')
        for i in range(0, len(bulb)):
            wf.write(f'{bulb[i][0]} {bulb[i][1]}\n')

        wf.write(f'\n')

    wf.close()


# write logs
def logs_write(log_file_path, runs, run_log):
    with open(log_file_path, 'a+') as wf:
        wf.write(f"\nrun {runs + 1} \n")
        for i in range(0, len(run_log)):
            wf.write(f'{str(run_log[i].get("evaluation"))}'
                     f'   {str(run_log[i].get("average_fitness"))} '
                     f'   {str(run_log[i].get("best_fitness"))}  \n'
                     )
    wf.close()


# insert a row of log data
def log_row(population_pool, number_evaluation):
    best_fitness = sorted(population_pool, key=itemgetter('evaluation_fitness'), reverse=True)[0]
    fitness_total = 0
    for i in range(0, len(population_pool)):
        fitness_total += population_pool[i].get('evaluation_fitness')

    log = {
        "evaluation": number_evaluation,
        "average_fitness": format(fitness_total / len(population_pool), '.2f'),
        "best_fitness": best_fitness.get('evaluation_fitness'),
        "sum_fitness": fitness_total,
        "best_fit_data": best_fitness
    }

    return log


# truncation for survival
def truncation(population_pool, number_survival):
    pool = sorted(population_pool, key=itemgetter('evaluation_fitness'), reverse=True)[:number_survival]
    random.shuffle(pool)
    return pool