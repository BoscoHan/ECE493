from gym.envs.registration import register

register(
    id='MazeWorld-v0',
    entry_point='maze.maze_env:Maze'
)