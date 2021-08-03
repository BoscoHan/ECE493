import random
from gym.envs.registration import register 
import numpy as np
import time
import sys
import gym
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import tkinter

# debug
from pdb import set_trace as bp

UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells
origin = np.array([UNIT/2, UNIT/2])

_agentXY=[0,0]
_goalXY=[2,5]
_wall_shape=np.array([[2,2],[3, 2], [4, 2], 
                         [2,3], [2, 4], [2,6], [2, 7],
                         [3, 8], [4, 8], [5, 8]])
_pits=np.array([[5,2],[2,8]])

canvas = None
env = None

class Maze(gym.Env):
    def __init__(self, agentXY = _agentXY, goalXY= _goalXY, walls=_wall_shape,  pits=_pits):
        self.wallblocks = []
        self.pitblocks=[]

        # For gym
        self.action_space = gym.spaces.Discrete(4) # 4 possible directions for moving 
        self.observation_space = gym.spaces.Discrete(100) # number of possible positions on grid (10 * 10)

        top = tk.Tk()
        self.build_shape_maze(agentXY, goalXY, walls, pits, top)
        
        global env
        env = self

    def build_shape_maze(self,agentXY,goalXY, walls, pits, _top):
        global canvas
        canvas = tkinter.Canvas(_top, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        self.canvas = canvas

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        for x,y in walls:
            self.add_wall(canvas, x,y)
        for x,y in pits:
            self.add_pit(canvas, x,y)
        self.add_goal(canvas, goalXY[0],goalXY[1])
        self.add_agent(canvas, agentXY[0],agentXY[1])


        canvas.pack()
        # keeps UI going
        _top.mainloop()

    def add_wall(self, canvas, x, y):
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='black'))

    def add_pit(self, canvas, x, y):
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='blue'))

    def add_goal(self, canvas, x=4, y=4):
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    def add_agent(self, canvas, x=0, y=0):
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')


    def reset(self, canvas = canvas, value = 1, resetAgent=True):
        update(canvas)
        time.sleep(0.2)
        if(value == 0):
            return canvas.coords(self.agent)
        else:
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.agent = canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                origin[0] + 15, origin[1] + 15,
                fill='red')

            # return curr state:
            return canvas.coords(self.agent)


    def step(self, canvas, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        s_ = canvas.coords(self.agent)  # next state

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = canvas.coords(self.agent)

        return s_, reward, done



def update(canvas = canvas):
    for t in range(10):
        print("The value of t is", t)
        # s = env.reset()
        while True:
            # env.render()
            a = 1
            s, r, done = env.step(canvas, a)
            if done:
                break



if __name__ == '__main__':
    env = Maze()
    env.after(100, update)

    env.mainloop()


