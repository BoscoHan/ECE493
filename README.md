# Slime Volleyball Gym Environment - Final Assignment

Assignment code for course ECE 493 T25 at the University of Waterloo in Spring 2020.

**Due Date:** TBD: submitted as PDF and code to LEARN dropbox.

**Collaboration:** You can discuss solutions and help to work out the code. But each person *must do their own work*. All code and writing will be cross-checked against each other and against internet databases for cheating. 

Updates to code which will be useful for all or bugs in the provided code will be updated on gitlab and announced.

## Domain Description


SlimeVolleyGym is a simple gym environment for testing Reinforcement Leanring algorithms. Please refer to the original repo at [slimevolleygym](https://github.com/hardmaru/slimevolleygym) to get more information about this environment. 


## Assignment Requirements

This assignment will have a written component and a programming component.
Clone the slimevolleygym environment locally and run the code looking at the implemtation of the sample algorithm.
Implementation of PPO (using stable baselines package) is given in the codebase. Refer to file ``slimevolleygym/training_scripts/train_ppo.py`` to see an example of how this training happens, here the training is against a baseline RNN policy that controls the left agent (This baseline policy is already learned and is not learning anymore) . Your task is to use an RL algorithm to control the right agent.   

Similarly this assignment will expect you to train several other RL algorithms that we have listed below. You need not implement these RL algorithms by hand. We suggest that you use the [stable baselines] (https://github.com/hill-a/stable-baselines) package as done in the example ``train_ppo.py`` script. Feel free to play with the hyperparameters to arrive at the best one. In the report highlight the steps you tried to find the best hyperparameter for all the algorithms. 


Your task is to implement three algortihms on this domain.
- **(20%)** Implement DQN
- **(20%)** Implement A2C
- **(20%)** At least one other algorithm of your choice or own design. 
- **(40%)** Report : Write a short report on the problem and the results of your three algorithms. The report should be submited on LEARN as a pdf. 
    - Describing each algorithm you used, define the states, actions, dynamics. Define the mathematical formulation of your algorithm, show the Bellman updates for you use.
    - Some quantitative analysis of the results, a default plot for comparing all algorithms is given. You can do more than that.
    - Some qualitative analysis of why one algorithm works well in each case, what you noticed along the way.


### Evaluation
You will also submit your code to LEARN and grading will be carried out using a combination of automated and manual grading.
We will look at your definition and implmentation which should match the description in the document.




## Installation (guide for Linux and OSX, for Windows please look online for instructions for the respective packages indicated here)

You need a version of python that is > 3.5 and < 3.8 installed in your machine. If you have other versions installed, you can have parallel installations, but use the python between 3.5 and 3.8.   

First clone the repo, 

```
git clone https://github.com/hardmaru/slimevolleygym.git
cd slimevolleygym
sudo pip3 install -e .
pip3 install opencv-python
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev (for ubuntu)
brew install cmake openmpi (for OSX)
pip3 install 'stable-baselines[mpi]'
pip3 install tensorflow==1.15
pip3 install gym
```

Note that tensorflow < 1.15 is needed for stable baselines as it does not support tensorflow 2.0 yet.

After cloning and installing all the packages, you can run the ``test_state.py`` file to play the game manually against the baseline agent. You can use the arrow keys and you will control the right agent. 

## Environments

There are two types of environments: state-space observation or pixel observations for the slimevolleygym environment. We will only be using the state-space observation for this Assignment. This environment is labelled as `SlimeVolley-v0`. Look at the original repo and familiarize yourself with the state space, action space and the reward function. 


This assignment will focus on the single-agent version where you will train an agent to compete against the baseline agent. You can also use the multi-agent version to compare the performances of your agents against those of your classmates. This is optional and is only for fun. Several baselines that you can try are mentioned in the ``Training.md`` file. To run the multiagent version run the following command 

```
python3 eval_agents.py --left ppo --right cma --render
``` 


Look at the ``eval_agents.py`` file to understand how the multiagent competition happens. You can replace the algorithms ``ppo`` and ``cma`` with the policies of you and your classmates to have a fun comparison. 

 

