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
Implementation of PPO (using stable baselines package) is given in the codebase.
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




## Installation (guide for Linux and OSX, for Windows please look online for instructions)

First clone the repo, 

```
git clone https://github.com/hardmaru/slimevolleygym.git
cd slimevolleygym
sudo pip3 install -e .
pip3 install opencv-python
pip3 install stable_baselines
pip3 install gym
```
After cloning and installing all the packages, you can run the ``test_state.py`` file to play the game manually against the baseline agent. You can use the arrow keys and you will control the right agent. 

## Environments

There are two types of environments: state-space observation or pixel observations for the slimevolleygym environment. We will only be using the state-space observation for this Assignment. This environment is labelled as `SlimeVolleyPixel-v0`. Look at the original repo and familiarize yourself with the state space, action space and the reward function. 


This assignment will focus on the single-agent version where you will train an agent to compete against the baseline agent. You can also use the multi-agent version to compare the performances of your agents against those of your classmates. This is optional and is only for fun. 

 

