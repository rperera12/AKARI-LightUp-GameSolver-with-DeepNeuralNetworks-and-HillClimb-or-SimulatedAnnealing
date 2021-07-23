# AKARI/Light Up Game Solver using Deep Neural Networks, Hill Climb, and Simulated Annealing Algorithms

Should you find this repository as a useful tool for research or application, please kindly cite the original article [Optimized and autonomous machine learning framework for characterizing pores, particles, grains and grain boundaries in microstructural images](https://arxiv.org/abs/2101.06474)

      @misc{sun2021shedding,
	      title={Shedding some light on Light Up with Artificial Intelligence}, 
	      author={Libo Sun and James Browning and Roberto Perera},
	      year={2021},
	      eprint={2107.10429},
	      archivePrefix={arXiv},
	      primaryClass={cs.AI}
	} 

We chose to focus on the [Light Up puzzle](https://en.wikipedia.org/wiki/Light_Up_(puzzle)), and solved it using hill climbing, simulated annealing, and a deep neural network.


#       running codes  	#

	cd codes
	python3 source/main.py puzzles/03.txt source/default.cfg

# 	code running example on a vidoe	#

	https://youtu.be/o6Wm-owLuqI


#       configuration:  	default.cfg			#

1.	optimizied initialization?  true or false	

		unique_optimize = true / false

2.	enable simulated annealing?  true or false	 

		annealing = true /false

3.	set simulated annealing temperature	

		annealing_temp = 0.5

4.	set random seed, time or an integer	 

		random_seed = "time"

5.	internal parameters for algorithm, test purpose   

		parameters_for_Black = 4

6.	set the number of runs  at a time    

		number_runs: 10

7.	set the number of evaluations/neighbors for a single run    

		termination_evaluation =  1000

8.	set the log saving path and file name apprex 	

		log_path = ./logs/test_
		solution_path = ./solutions/test


# Board Legend #

	CELL_BLACK_ZERO = 0
	CELL_BLACK_ONE = 1
	CELL_BLACK_TWO = 2
	CELL_BLACK_THREE = 3
	CELL_BLACK_FOUR = 4
	CELL_BLACK_FIVE = 5
	CELL_BULB_ZERO = 9 (This value indicates that a bulb cannot be placed adjacent to this cell.)
	CELL_EMPTY = 6 (This value indicates an empty white cell.)
	CELL_BULB = 7 (This value indicates a white cell with a bulb inside it.)
	CELL_LIGHT = 8 (This value indicates that a cell is lit up by a bulb.)


#       Neural Network section  	#


1 - Dependencies:
	For simplicity you can copy paste the following modules and packages.
	If missing some you can install on your PC.

	from __future__ import print_function, division
	import torch
	import torch.nn as nn
	import torchvision
	from torchvision import datasets, transforms, utils
	from torch import nn, optim
	from torch.autograd import Variable
	from torch.utils.data import Dataset, DataLoader
	import torch.nn.functional as F
			
	import cv2
	import os
	import numpy as np
	import time
	import matplotlib.pyplot as plt
	from skimage import transform

2 - Training Models:

	a) command: 
		cd codes
		python3 trainer.py
	b) changes: 
		various CNN models can be used to trained. 
		The models are available in models.py; 
		feel free to design your own :)
	c) pretrained models: Already trained weights can be found in codes/logs/Pretrained_Models/
		(i)  HillClimb_trained.pt - 
			HillClimb version using ConvNet3 in models.py
		(ii) Annealing_Conv4_Akari.pt - 
			Simulated Annealing version using ConvNet4 
			including: dropout regularization) in models.py

3 - Testing Models:

	a) command: 
		cd codes 
		python3 test_neuralnets.py
	b) changes: 
		you can modify the pretrained model from codes/logs/Pretrained_Models/
		youi can also test your own trained models
		
		Note: Make sure to load the corresponding model from models.py
			 - Refer to the section above for more details.

4 - Visualizing the Boards: 

	a) Board Function -    	
		To visualize each board, a class can be found on boards.py. 
	b) Generating boards - 
		To generate them, a function was developed to run boards.py in the utils.py code
			 
	** NOTE: 
		You don't need to implement anything new as the board.py 
		and utils.py are called directly in trainer.py 
		and test_neuralnets.py - ENJOY.
