# Project:      “Shedding Some Light on the Light Up Puzzle with Artificial Intelligence”
# Program name: main.py
# Author:       James Browning (jlb0181@auburn.edu), Robert Perera (rzp0063@auburn.edu),
# and Libo Sun (lzs0101@auburn.edu)
# Date created: October 8th, 2020
# Purpose:      Use techniques Artificial Intelligence to solve the light up puzzle.


# import from project functions
import multiprocessing
from multiprocessing import Process
from copy import deepcopy
import time

import initial as init
import hill
import logs


# --------imports from system------
import sys
import numpy as np



# read arguments
def config_argv():
    if len(sys.argv) < 3:
        print("Error")
        sys.exit(0)
    # reading arguments from arg vector
    problem_filepath = sys.argv[1]
    config_filepath = sys.argv[2]
    print(problem_filepath, config_filepath)


# a single run of hill climbing
def single_run(run, game_map: hill.Map, configuration: init.Config, result):
    # print(f'run:....{run}')

    fitness = hill.evaluate_puzzle_map(game_map)
    # print(f'Initial fitness: {fitness}')

    logs_running = [[0, fitness]]

    terminate = False
    number_evals = 0
    running_board = deepcopy(game_map)

    local_optima = running_board
    while not terminate:
        number_evals += 30

        running_board = hill.hill_climb(running_board, configuration)
        # print(f'Evaluations: {number_evals}   The current fitness: {running_board.fitness}')

        # running_board.log_update(number_evals, running_board.fitness)
        logs_running.append([number_evals, running_board.fitness])

        if number_evals > configuration.evaluation_number:
            # print("Terminated by number of evals")
            terminate = True
        if running_board.fitness == 100:
            # print("Terminated by 100 fitness")
            terminate = True

        if local_optima.fitness < running_board.fitness:
            local_optima = running_board
            
    local_optima.result_log = logs_running
    result[run] = local_optima

    return local_optima


# Print the maps
def print_maps(game_map):
    print("initial board.....")
    for i in range(0, game_map.column):
        print(game_map.board[game_map.column - i - 1])

    print("optimized board.....")
    for i in range(0, game_map.column):
        print(game_map.optimized_board[game_map.column - i - 1])

    print("insert bulbs for black adjacency")
    for i in range(0, game_map.column):
        print(game_map.board_running[game_map.column - i - 1])



# Input: configurations of the project, the maps(three maps)
# Output: global fitness and overall running time 
def perform_experiment(configurations, game_map) -> float:
    # starting multiple processing.......
    start = time.perf_counter()
    processes = []
    manager = multiprocessing.Manager()
    all_logs = manager.dict()
    for i in range(0, configurations.runs):
        processes.append([])
        processes[i] = Process(target=single_run, args=(i, game_map, configurations, all_logs))
        processes[i].start()

    for i in range(0, configurations.runs):
        processes[i].join()

    global_optima = all_logs[0]
    for i in range(0, len(all_logs)):
        # print(f'run: {i} {all_logs[i].fitness}')
        # logs.logs_write(configurations.log_file, i, all_logs[i].result_log)
        if global_optima.fitness < all_logs[i].fitness:
            global_optima = all_logs[i]
            

    print(f'global optimal fitness: {global_optima.fitness}')
    # print out the global optimal
    board = global_optima.optimized_board
    board = hill.insert_bulbs(board, global_optima.bulb_running)
    hill.check_bulb_shining(board, global_optima.row, global_optima.column)
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')
    return round((finish - start), 2)

    
# write the logs for experiment time    
def record_time(config, exp_time):
    t = np.array(exp_time)
    with open(config.log_file, 'a+') as wf:
        wf.write(f"Map file:     {config.map_file_name} \n")
        wf.write(f"Config file:  {config.config_file_name} \n")
        wf.write(f"--Puzzle size: {config.get_puzzle_size()} \n")
        wf.write(f"--Black Constraints: {config.black_constraints} \n")
        wf.write(f"--Simulated Annealing: {config.annealing} \n")
        wf.write(f"--Moving Action: {config.moving_action} \n")
        wf.write(f"\nExperiment time: (seconds) \n")
        for x in t:
            wf.write(f"{x}\n")        
    wf.close()   


# Main function 
# execute example: python3 source/main.py puzzles/03.txt source/default.cfg
def main():
    if len(sys.argv) < 3:
        print("Error")
        sys.exit(0)

    if len(sys.argv) == 3 or len(sys.argv) > 3:

        # reading arguments from arg vector
        map_file = sys.argv[1]
        config_file = sys.argv[2]

        configurations = init.Config(config_file)
        configurations.set_filename(map_file)
        
        configurations.print_initialization()
        game_map = hill.Map(map_file, configurations)
        configurations.set_puzzle_size(game_map.row, game_map.column)
       

        game_map.set_start_map()
        
        print_maps(game_map)

        experiment_time = []
        for i in range(30):
            print(f'Experiment {i}: ...')
            experiment_time.append(perform_experiment(configurations, game_map))
        record_time(configurations, experiment_time)
        


if __name__ == "__main__":
    main()
