# ------- all imports ------
import math
import sys
import random
import statistics
import numpy
from random import randint
from copy import deepcopy
from datetime import datetime

from operator import itemgetter

import json

from numpy.lib import utils


import setup


import draw

# -------- constants  --------


# read config
def config_argv():
    if len(sys.argv) != 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    config_filepath = sys.argv[1]
    problem_filepath = sys.argv[2]
    
    print(problem_filepath, config_filepath)


# ======== Programming Starts here:  Main Function =========
def main():
    
    if len(sys.argv) != 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    config_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    print(problem_file, config_file)

    ea = setup.Board(config_file, problem_file)

    for i in range(len(ea.board_grids)):
        print(ea.board_grids[len(ea.board_grids) - i - 1])

    
    ea.set_black_zero()
    ea.set_unique_bulbs()


    print("After Insert unique bulbs.....")
    for i in range(len(ea.board_grids)):
        print(ea.board_grids[len(ea.board_grids) - i - 1])

    
    board = draw.DrawBoard(ea.board_grids)
    board.show()
        
       
    

    exit(0)

    number_cols = puzzle_initial[0][0]
    number_rows = puzzle_initial[0][1]

    # create an initial map by col/row
    puzzle_map = initialize_map(number_cols, number_rows)

    # update myMap by loading the config
    puzzle_map = load_map_data(puzzle_map, puzzle_initial)

    for i in range(0, len(puzzle_map)):
        print(puzzle_map[len(puzzle_map) - i - 1])

    # $$$$$$$  config begins here  $$$$$$$$$$$$$$$$$
    mu_size = puzzle_config["mu_size"]
    lambda_size = puzzle_config["lambda_size"]

    termination_constant = {"number_evaluation": puzzle_config["termination_evaluation"],
                            "no_change": puzzle_config["termination_no_change"]
                            }
    # $$$$$$$  config ends here  $$$$$$$$$$$$$$$$$

    # implement constraints or not
    if puzzle_config["black_cell_constraints"]:
        puzzle_constraints = initialize_validation_map(puzzle_map, number_rows, number_cols)
        print("Printing after initializing by unique bulbs...")
        for i in range(0, len(puzzle_constraints)):
            print(puzzle_constraints[len(puzzle_constraints) - i - 1])
        net_white_cells = check_net_cells(puzzle_constraints, number_rows, number_cols)
        puzzle_map = puzzle_constraints
    else:
        net_white_cells = check_net_cells(puzzle_map, number_rows, number_cols)

    solution = []

    for runs in range(0, puzzle_config["number_runs"]):

        run_log = []
        mutation_rate = []

        # initialize population pool of mu - parents
        initial_mu = create_population_pool(net_white_cells, mu_size)

        population_mu = []
        for i in range(len(initial_mu)):
            evaluations = evaluate_puzzle_map(puzzle_map,
                                              black_cells=puzzle_initial,
                                              bulb_cells=initial_mu[i],
                                              config=puzzle_config)
            population_mu.append(evaluations)

        best_mu = sorted(population_mu, key=itemgetter('evaluation_fitness'), reverse=True)[0]
        number_evaluation = mu_size
        run_log.append(log_row(population_mu, number_evaluation))

        # prepare variables for EA loop
        termination = False

        best_fitness_run = best_mu
        no_change_best = 0
        # start to EA loop
        while not termination:

            parents_pool = parent_selection(population_mu, lambda_size,
                                            puzzle_config["parent_selection"],
                                            puzzle_config["k_tournament_parents"])

            population_lambda = generate_offspring(net_white_cells, lambda_size,
                                                   parents_pool, puzzle_map, puzzle_initial,
                                                   puzzle_config, mutation_rate)

            best_fitness_current_population = sorted(population_mu + population_lambda,
                                                     key=itemgetter('evaluation_fitness'),
                                                     reverse=True)[0]

            population_mu = survival_selection(puzzle_config, population_mu, population_lambda)

            if puzzle_config["self_adaptive"]:
                number_evaluation += lambda_size

            number_evaluation += lambda_size

            run_log.append(log_row(population_mu, number_evaluation))

            if best_fitness_run.get('evaluation_fitness') < best_fitness_current_population.get('evaluation_fitness'):
                best_fitness_run = best_fitness_current_population
                no_change_best = 0
            else:
                no_change_best += lambda_size
                if no_change_best >= puzzle_config["termination_no_change"]:
                    termination = True

            if number_evaluation >= termination_constant["number_evaluation"]:
                termination = True
        # print(mutation_rate)

        # write running log into file

        logs_write(log_file_path, runs, run_log)

        print(f"run: {runs + 1}")
        print(best_fitness_run)

        if best_fitness_run.get("evaluation_fitness"):
            if not solution:
                solution = best_fitness_run
            elif solution.get("evaluation_fitness") < best_fitness_run.get("evaluation_fitness"):
                solution = best_fitness_run

    print(f"the best overall: {solution}")
    if solution:
        solution = insert_bulbs(puzzle_map, solution.get("bulbs"))
        check_bulb_shining(solution, number_rows, number_cols)
        for i in range(0, len(solution)):
            print(solution[len(solution) - 1 - i])
        write_solution(puzzle_initial, solution, puzzle_config["solution_file"])

    return


if __name__ == "__main__":
    main()
