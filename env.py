import numpy as np
import copy

# for rendering
AGENT_PLOT = 'A'
GOAL_PLOT = 'G'
GRID_PLOT = '_'

# for initialize state
WORLD_WIDTH = 10
GOAL_POSITION = np.zeros(WORLD_WIDTH)
AGENTS_INITIAL_POSIITON = np.zeros(WORLD_WIDTH)
MOST_LEFT_POSITION = np.zeros(WORLD_WIDTH)
GOAL_POSITION[-1] = 1
AGENTS_INITIAL_POSIITON[4] = 1
MOST_LEFT_POSITION[0] = 1
# GOAL_POSITION=np.array([0,0,0,0,0,0,0,0,0,1])
# AGENTS_INITIAL_POSIITON=np.array([0,0,0,0,1,0,0,0,0,0])
# MOST_LEFT_POSITION=np.array([1,0,0,0,0,0,0,0,0,0])


class Find_Goal_Environment():
    def __init__(self):
        self.goal_position = copy.deepcopy(GOAL_POSITION)
        self.agent_position = copy.deepcopy(AGENTS_INITIAL_POSIITON)
        self.most_left_position = copy.deepcopy(MOST_LEFT_POSITION)
        self.world_width = copy.deepcopy(WORLD_WIDTH)
        self.world = np.chararray(self.world_width)

    def step(self, action):
        # go right
        if action == 1:
            pre_position = np.argmax(self.agent_position)
            cur_position = np.argmax(self.agent_position)+1
            self.agent_position[pre_position] = 0
            self.agent_position[cur_position] = 1
        # go left
        elif action == 0:
            # if you are already at the most left grid, stay still
            if np.array_equal(self.agent_position, self.most_left_position):
                pass
            # go left
            else:
                pre_position = np.argmax(self.agent_position)
                cur_position = np.argmax(self.agent_position)-1
                self.agent_position[pre_position] = 0
                self.agent_position[cur_position] = 1
        else:
            print('action value error')
        # check if goal is reached
        if np.array_equal(self.agent_position, self.goal_position):
            reward = 100
            terminated = True
        else:
            reward = -1
            terminated = False

        return reward, terminated

    def get_state(self):
        return tuple(self.agent_position)

    def reset(self):
        self.agent_position = copy.deepcopy(AGENTS_INITIAL_POSIITON)

    def render(self):
        self.world = np.chararray(self.world_width)
        self.world[:] = GRID_PLOT
        self.world[np.argmax(self.agent_position)] = AGENT_PLOT
        self.world[np.argmax(self.goal_position)] = GOAL_PLOT
        print(self.world)
