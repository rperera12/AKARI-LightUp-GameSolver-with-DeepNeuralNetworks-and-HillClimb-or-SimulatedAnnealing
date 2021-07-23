# Project:      “Shedding Some Light on the Light Up Puzzle with Artificial Intelligence”
# Program name: main.py
# Author:       James Browning (jlb0181@auburn.edu), Robert Perera (rzp0063@auburn.edu),
# and Libo Sun (lzs0101@auburn.edu)
# Date created: October 8th, 2020
# Purpose:      Use techniques Artificial Intelligence to solve the light up puzzle.


# import from project functions
from copy import deepcopy

import initial as init
import hill
import logs
from board import GameBoard
from utils import Generate_Board, get_img_path, get_play_path, generate_dataset

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

        # config_file = "default2.cfg"
        # map_file = "./maps/map2.txt"
        configurations = init.Config(config_file)
        configurations.set_filename(map_file)

        print(configurations.annealing)
        print(configurations.black_constraints)
        print(configurations.log_file)
        print(configurations.solution_file)

        game_map = hill.Map(map_file, configurations)

        # Define Images Directory to locate board Pieces
        pieces_dir = get_img_path()

        board_i = np.zeros([game_map.row, game_map.column], dtype=np.uint8)
        board_opt = np.zeros([game_map.row, game_map.column], dtype=np.uint8)

        print("initial board.....")

        for i in range(0, game_map.column):
            board_i[i, :] = game_map.board[game_map.column - i - 1]
            print(game_map.board[game_map.column - i - 1])

        print("optimized board.....")

        for i in range(0, game_map.column):
            board_opt[i, :] = game_map.optimized_board[game_map.column - i - 1]
            print(game_map.optimized_board[game_map.column - i - 1])

            # Check Length of args - If more than 3 provided check if display board of make dataset
        if len(sys.argv) > 3:
            # Generate Display for Boards if argv[3] is --display
            if "--display" in sys.argv:
                Generate_Board(pieces_dir, board_i)
                Generate_Board(pieces_dir, board_opt)

            # Generate cvs files for dataset if argv[3] is --dataset
            if "--dataset" in sys.argv:
                val = input("Enter your the size of training data to generate/store: ")
                vals = int(val)
                generate_dataset(vals)

        # song = get_play_path()
        # playsound(song)

        exit(0)


if __name__ == "__main__":
    main()
