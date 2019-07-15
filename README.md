
# MADDPG

### Introduction

This experiment implements the Multi Agent Deep Deterministic Policy Gradient algorithm to train a two independent agents to learn how to keep passing the ball back and fort (rallying) inside the unity ML-Agents virtual [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. In this environment, two paddles move within their side of the tennis court in an attempt to hit the ball back to the opposite agent.

![Trained Agent](/images/trained_maddpg.gif)

### Rewards

+0.1 To agent when hitting ball over net.
-0.1 To agent who let ball hit their ground, or hit ball out of bounds.

### State Space

The observation space consists of 8 variables corresponding to position and velocity of the ball and agents racket

### Action Space

Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.

### Solving the Environment
The environment is considered solved when the average score of all agents in the environment (in this case 2) for a period of 100 episodes is 0.5 or above, with a max score of 2.5 .


### Setup

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name ddpg python=3.6
	source activate ddpg
	```
	- __Windows__: 
	```bash
	conda create --name ddpg python=3.6 
	activate ddpg
	```

2. Clone the repository and install dependencies.
```bash
git clone https://github.com/djbyrne/MADDPG.git
cd MADDPG
pip install .
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
        
4. Finally, run the setup.py file to install all dependencies for this project


### Instructions

All code for this project is contained in the MADDPG.ipynb notebook. As such you just need to run the cells in order to see the results.
