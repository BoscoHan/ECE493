# Slime Volleyball Gym Environment - Assignment 3

Assignment code for course ECE 493 T25 at the University of Waterloo in Spring 2020.

**Due Date:** Thursday Aug 5, 2021 by 11:59pm: submitted as PDF to Crowdmark and code to LEARN group dropbox.

**Collaboration:** You can discuss solutions and help to work out the code. The assignment can be done alone or as a pair, pairs do not need to be the same as for assignment 2.
All code and writing will be cross-checked against each other and against internet databases for cheating. There is some hints at the end about playing your algorithms head to head against other students, this is optional but encouraged.

Updates to code which will be useful for all or bugs in the provided code will be updated on gitlab and announced.

## Domains for this Assignment
This assignment will use two domains to test out Deep RL algorithms. 
1. `Maze World` from Assignment 2 (See https://git.uwaterloo.ca/mcrowley/ece493t25-assignment2-spring2020)
2. `SlimeVolleyGym`: this is a simple gym environment for testing Reinforcement Learning algorithms. Refer to the original repo at [slimevolleygym](https://github.com/hardmaru/slimevolleygym) to get more information about this environment. 


## Assignment Requirements

This assignment will have a written component and a programming component.
Clone the slimevolleygym environment locally and run the code looking at the implemtation of the sample algorithm.
Implementation of PPO (using the stable baselines package) is given in the codebase. Refer to file ``/training_scripts/train_ppo.py`` to see an example of how this training happens, here the training is against a baseline RNN policy that controls the left agent (This baseline policy is already learned and is not learning anymore) . Your task is to use an RL algorithm to control the right agent.

Similarly, this assignment will expect you to train several other RL algorithms that we have listed below. You need not implement these RL algorithms by hand. We suggest that you use the [stable baselines] (https://github.com/hill-a/stable-baselines) package as done in the example ``train_ppo.py`` script. Feel free to play with the hyperparameters to arrive at the best one. In the report highlight the steps you tried to find the best hyperparameter for all the algorithms. 


Your task is to run or implement some Deep RL algortihms on this domain. There are two options for how to do this, using DeepRL libraries or coding it up yourself:
- OPTION 1:
    - **(20%)** Implement DQN using the stable baselines package and test on both environments
    - **(20%)** Implement A2C using the stable baselines package and test on both environments
    - **(20%)** At least one other algorithm of your choice using the stable baselines package and run on both environments. 
- OPTION 2:
    - **(60%)** Implement A2C from scratch using your own defined Deep Neural Networks and test on both environments
        - grading will be based on : design of networks; correct definitions of value functions, rewards, gradients, etc; code runs on both environments; performance is reasonably good compared to the baselines version (but it does *not* need to be equivalent to it)
- EITHER OPTION: 
    - **(40%)** Report : Write a short report on the problem and the results of your algorithms. The report should be submited on crowdmark as a pdf. 
        - Describing each algorithm you used, define the states, actions, dynamics. Define the mathematical formulation of your algorithm, show the Bellman updates for you use.
        - Some quantitative analysis of the results, a default plot for comparing all algorithms is given. You can do more than that.
        - Clearly mention the hyper-parameters used and the steps that you took to arrive at this value. 
        - Some qualitative analysis of why one algorithm works well in each case, and what you noticed along the way.
        - Note: if it is more convenient, you can report all of the results for one environment first, then all of the results for the second environment.


### Evaluation
You will also submit your code to LEARN and grading will be carried out using a combination of automated and manual grading.
We will look at your definition and implmentation which should match the description in the document.



## Installation (guide for Linux and OSX, for Windows please look online for instructions for the respective packages indicated here)
We have prepared for you a conda env file. The code use a specific version of python (3.7) and a specific version of TF (1.5) so a virtual environment is recommended. 

### Setup conda
Please follow instructions [here](https://docs.anaconda.com/anaconda/install/)

### Setup environment

```
#clone the repo
git clone https://git.uwaterloo.ca/mcrowley/ece493finalassignment.git
cd ece493finalassignment

#setup and activate environment
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev (for ubuntu)
brew install cmake openmpi (for OSX)
conda env create -f env493a3.yml
conda activate 493a3
pip3 install -e .

#test
python3 eval_agents.py --left ppo --right cma --render
```
After cloning and installing all the packages, you can run the ``test_state.py`` file to play the game manually against the baseline agent. You can use the arrow keys and you will control the right agent. 

Note: Stable baselines is just a suggestion. There are other RL packages out with different deep learning libraries that can also be tried. 

## Required and Extra Environments for slimevolleygym

There are two types of environments for the slimevolleygym environment: 
- **state-space observation** 
- pixel observations

For this assignment, you are **only required to use the state-space observation**. This environment is labelled as `SlimeVolley-v0`. Look at the original repo and familiarize yourself with the state space, action space and the reward function. If you are interested in going futher, you can explore using the pixel observations environment to make the problem harder, and more generalizable to different games.


This assignment will focus on the single-agent version where you will train an agent to compete against the baseline agent. You can also use the multi-agent version to compare the performances of your agents against those of your classmates. This is optional and is only for fun. Several baselines that you can try are mentioned in the ``Training.md`` file. To run the multiagent version run the following command 

```
python3 eval_agents.py --left ppo --right cma --render
```


Look at the ``eval_agents.py`` file to understand how the multiagent competition happens. You can replace the algorithms ``ppo`` and ``cma`` with the policies of you and your classmates to have a fun comparison. 

 

