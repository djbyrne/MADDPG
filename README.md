
# MADDPG

### Introduction

This experiment implements the Multi Agent Deep Deterministic Policy Gradient algorithm to train a two independent agents to learn how to keep passing the ball back and fort (rallying) inside the unity ML-Agents virtual [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. In this environment, two paddles move within their side of the tennis court in an attempt to hit the ball back to the opposite agent.

![Trained Agent][https://raw.githubusercontent.com/djbyrne/MADDPG/master/images/trained_maddpg.gif?token=AEoVZYeqUGUA6821_4WQLRgJy7Abawfdks5ca5lywA%3D%3D]

### Rewards

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### State Space

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

### Action Space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment
The environment is considered solved when the average score of all agents in the environment (in this case 20) for a period of 100 episodes is 30 or above.

### Setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the project directory and change the environment path in the notebook

3. Finally, run the setup.py file to install all dependencies for this project

### Instructions

All code for this project is contained in the DDPG_Reacher.ipynb notebook. As such you just need to run the cells in order to see the results.
