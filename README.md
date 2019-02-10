# Maze Reinforcement Learning - README

## Installation
This code was written for Python 3 and requires the following packages: Numpy, Math, Time and Scipy.

## Overview
This repository contains the code used to solve the maze reinforcement learning problem described [here](https://medium.com/@gkhayes/the-other-type-of-machine-learning-97ab81306ce9). It uses the Q-learning algorithm with an epsilon-greedy exploration strategy.

## File Descriptions
The Python file intro_to_rl_example.py contains the code used to define and solve the reinforcement learning problem. 

All other files (mdp, util and error) are modified version of files contained in the MDPToolbox Python package, and are used to define the Q-Learning function. Modification of these files was necessary in order to redefine the exploration strategy used to being an epsilon greedy strategy.

To run this code, run intro_to_rl_example.py at either the command line or via an IDE, such as Spyder.

## Licensing, Authors, Acknowledgements
The code contained in mdp.py, util.py and error.py was primarily written by Steven Cordwell as part of the MDPToolbox Python package. The original version can be found [here](https://github.com/sawcordwell/pymdptoolbox).

All other code was written by Genevieve Hayes and may be used freely with acknowledgement.
