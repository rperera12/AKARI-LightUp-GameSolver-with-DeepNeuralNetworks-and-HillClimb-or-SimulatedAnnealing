# Project:      COMP 6600/6606 Final Project: “Shedding Some Light on the Light Up Puzzle with Artificial Intelligence”
# Program name: main.py
# Author:       James Browning (jlb0181@auburn.edu), Robert Perera (rzp0063@auburn.edu),
# and Libo Sun (lzs0101@auburn.edu)
# Date created: October 8th, 2020
# Purpose:      Use techniques learned in COMP 6600/6606 Artificial Intelligence to solve the light up puzzle.


# import from project functions
import multiprocessing
from multiprocessing import Process
from copy import deepcopy
import time

import initial as init
import hill
import logs
# from board import GameBoard
# from utils import Generate_Board, get_img_path, get_play_path, generate_dataset

# --------imports from system------
import os
import sys
from operator import itemgetter
import numpy as np
# from playsound import playsound


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
    print(f'run:....{run}')

    fitness = hill.evaluate_puzzle_map(game_map)
    print(f'Initial fitness: {fitness}')

    # game_map.log_update(0, fitness)
    logs_running = [[0, fitness]]

    terminate = False
    number_evals = 0
    running_board = deepcopy(game_map)
    non_fitness_improvement = 0

    local_optima = running_board
    while not terminate:
        number_evals += 30

        running_board = hill.hill_climb(running_board, configuration)
        print(f'Evaluations: {number_evals}   The current fitness: {running_board.fitness}')

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
            non_fitness_improvement = 0
        else:
            non_fitness_improvement += 1

        # if 100 iteration has not improved the fitness, then restart
        # if non_fitness_improvement > 100:
        #     non_fitness_improvement = 0
        #     print("No improvement 100 iterations, restarting...")
        #     running_board = deepcopy(game_map)
    local_optima.result_log = logs_running
    result[run] = local_optima

    return local_optima


# ======== ========= =========
# ======== Main body =========
# ======== ========= =========
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

        # Define Images Directory to locate board Pieces
        # pieces_dir = get_img_path()
        #
        # board_i = np.zeros([game_map.row, game_map.column], dtype=np.uint8)
        # board_opt = np.zeros([game_map.row, game_map.column], dtype=np.uint8)

        print("initial board.....")
        for i in range(0, game_map.column):
            # board_i[i, :] = game_map.board[game_map.column - i - 1]
            print(game_map.board[game_map.column - i - 1])

        print("optimized board.....")
        for i in range(0, game_map.column):
            # board_opt[i, :] = game_map.optimized_board[game_map.column - i - 1]
            print(game_map.optimized_board[game_map.column - i - 1])

        game_map.set_start_map()
        print("insert bulbs for black adjacency")
        for i in range(0, game_map.column):
            print(game_map.board_running[game_map.column - i - 1])

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
            print(f'run: {i}')
            logs.logs_write(configurations.log_file, i, all_logs[i].result_log)
            if global_optima.fitness < all_logs[i].fitness:
                global_optima = all_logs[i]
            for j in range(0, len(all_logs[i].result_log)):
                print(all_logs[i].result_log[j])

        print(f'global optimal fitness: {global_optima.fitness}')
        # print out the global optimal
        board = global_optima.optimized_board
        board = hill.insert_bulbs(board, global_optima.bulb_running)
        hill.check_bulb_shining(board, global_optima.row, global_optima.column)
        for i in range(0, global_optima.row):
            print(board[i])

        finish = time.perf_counter()
        print(f'Finished in {round(finish - start, 2)} second(s)')

        exit(0)

        # a single run for the puzzle
        # return a local optima
        # local_optima = single_run(game_map, configurations)

            # Check Length of args - If more than 3 provided check if display board of make dataset
        if len(sys.argv) > 3:
            # Generate Display for Boards if argv[3] is --display
            # if "--display" in sys.argv:
            #     Generate_Board(pieces_dir, board_i)
            #     Generate_Board(pieces_dir, board_opt)

            # Generate cvs files for dataset if argv[3] is --dataset
            if "--dataset" in sys.argv:
                val = input("Enter your the size of training data to generate/store: ")
                vals = int(val)
                # generate_dataset(vals)

        # song = get_play_path()
        # playsound(song)

        exit(0)


if __name__ == "__main__":
    main()
