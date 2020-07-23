import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import copy
import random
import math
import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as mpatches
# from matplotlib.patches import Circle, Rectangle
# import mpl_toolkits.mplot3d.art3d as Art3d
# from matplotlib.widgets import Button
# from matplotlib.widgets import CheckButtons
# import matplotlib.path as mpath

# auv's max speed (unit: m/s)
AUV_MAX_V = 2.0
AUV_MIN_V = 0.1
# auv's max angular velocity (unit: rad/s)
#   TODO: Currently, the track_way_point function has K_P == 1, so this is the range for w. Might change in the future?
AUV_MAX_W = np.pi/8

# shark's speed (unit: m/s)
SHARK_MIN_V = 0.5
SHARK_MAX_V = 1
SHARK_MAX_W = np.pi/8

RRT_PLANNER_FREQ = 10

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 3.0
FOLLOWING_RADIUS = 50.0

# the auv will receive an immediate negative reward if it is close to the obstacles
OBSTACLE_ZONE = 0.0
WALL_ZONE = 10.0

# constants for reward
R_FOUND_PATH = 300
R_CREATE_NODE = 0
R_INVALID_NODE = -1

REMOVE_CELL_WITH_MANY_NODES = True

# size of the observation space
# the coordinates of the observation space will be based on 
#   the ENV_SIZE and the inital position of auv and the shark
# (unit: m)
ENV_SIZE = 500.0

DEBUG = False
# if PLOT_3D = False, plot the 2d version
PLOT_3D = False

NODE_THRESHOLD = 30

"""
============================================================================

    Helper Functions

============================================================================
"""
def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians

    Note: 
        Because Python does not encourage importing files from the parent module, we have to place this angle wrap here. If we don't want to do this, we can possibly organize this so auv_env is in the parent folder?
    """
    if -np.pi <= ang <= np.pi:
        return ang
    elif ang > np.pi: 
        ang += (-2 * np.pi)
        return angle_wrap(ang)
    elif ang < -np.pi: 
        ang += (2 * np.pi)
        return angle_wrap(ang)

"""
============================================================================

    Class - Live3DGraph (for plotting)

============================================================================
"""
SIM_TIME_INTERVAL = 0.1

class Live3DGraph:
    def __init__(self, plot_3D):
        """
        Uses matplotlib to generate live 3D Graph while the simulator is running

        Able to draw the auv as well as multiple sharks
        """
        self.shark_array = []

        # array of pre-defined colors, 
        # so we can draw sharks with different colors
        self.colors = ['b', 'g', 'c', 'm', 'y', 'k']

        self.arrow_length_ratio = 0.1

        # initialize the 2d graph
        self.fig_2D = plt.figure(figsize = [13, 10])
        self.ax_2D = self.fig_2D.add_subplot(111)

        # an array of the labels that will appear in the legend
        # TODO: labels and legends still have minor bugs
        self.labels = ["auv"]

    def plot_obstacles_2D(self, obstacle_array, dangerous_zone_radius):
        """
        Plot obstacles as sphere based on location and size indicated by the "obstacle_array"

        Parameter - obstacle_array
            an array of motion_plan_states that represent the obstacles's
                position and size
        """
        for obs in obstacle_array:
            # TODO: fow now, plot circles instead of spheres to make plotting faster
            obstacle = Circle((obs.x,obs.y), radius = obs.size, color = '#000000', fill=False)
            self.ax_2D.add_patch(obstacle)

            if dangerous_zone_radius != 0.0:
                dangerous_zone = Circle((obs.x,obs.y), radius = obs.size + dangerous_zone_radius, color='#c42525', fill=False)
                self.ax_2D.add_patch(dangerous_zone)

"""
============================================================================

    Class - Motion Plan State

============================================================================
"""

"""a wrapper class to represent states for motion planning
    including x, y, z, theta, v, w, and time stamp"""
class Motion_plan_state:
    #class for motion planning

    def __init__(self,x,y,z=0,theta=0,v=0,w=0, traj_time_stamp=0, plan_time_stamp=0, size=0, rl_state_id = None):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.v = v #linear velocity
        self.w = w #angulr velocity
        self.traj_time_stamp = traj_time_stamp
        self.plan_time_stamp = plan_time_stamp
        self.size = size
        self.rl_state_id = rl_state_id

        self.parent = None
        self.path = []
        self.length = 0
        self.cost = []

    def __repr__(self):
        # # goal location in 2D
        # if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return ("MPS: [x=" + str(self.x) + ", y="  + str(self.y) + "]")
        # # goal location in 3D
        # elif self.theta == 0 and self.v == 0 and self.w == 0 and self.size == 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + "]"
        # # obstable location in 3D
        # elif self.size != 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return "MPS: [x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z) + ", size=" + str(self.size) + "]"
        # # location for Dubins Path in 2D
        # elif self.z ==0 and self.v == 0 and self.w == 0:
        #     return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", theta=" + str(self.theta) + ", trag_time=" + str(self.traj_time_stamp) + ", plan_time=" + str(self.plan_time_stamp) + ", state_id=" +  str(self.rl_state_id) + "]"
        # else: 
        return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) +\
            ", theta=" + str(self.theta)  + ", v=" + str(self.v) + ", w=" + str(self.w) +\
            ", trag_time=" + str(self.traj_time_stamp) +  ", plan_time="+  str(self.plan_time_stamp) + ", state_id=" +  str(self.rl_state_id) + "]"

    def __str__(self):
        # # goal location in 2D
        # if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return ("MPS: [x=" + str(self.x) + ", y="  + str(self.y) + "]")
        # # goal location in 3D
        # elif self.theta == 0 and self.v == 0 and self.w == 0 and self.size == 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + "]"
        # # obstable location in 3D
        # elif self.size != 0 and self.traj_time_stamp == 0 and self.plan_time_stamp == 0:
        #     return "MPS: [x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z) + ", size=" + str(self.size) + "]"
        # # location for Dubins Path in 2D
        # elif self.z ==0 and self.v == 0 and self.w == 0:
        #     return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", theta=" + str(self.theta) + ", trag_time=" + str(self.traj_time_stamp) + ", plan_time=" + str(self.plan_time_stamp) + ", state_id=" +  str(self.rl_state_id) + "]"
        # else: 
        return "MPS: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) +\
            ", theta=" + str(self.theta)  + ", v=" + str(self.v) + ", w=" + str(self.w) +\
            ", trag_time=" + str(self.traj_time_stamp) +  ", plan_time="+  str(self.plan_time_stamp) + ", state_id=" +  str(self.rl_state_id) + "]"

"""
============================================================================

    Class - Grid Cell RRT (Helper Class for Planner_RRT)

============================================================================
"""
class Grid_cell_RRT:
    def __init__(self, x, y, side_length = 1, num_of_subsections = 8):
        """
        Parameters:
            x - x coordinates of the bottom left corner of the grid cell
            y - y coordinates of the bottom left corner of the grid cell
            side_length - side_length of the square grid cell
            sub_sections - how many subsections (based on theta) should the grid cell be split into
                the subsection are created in counter-clock wise direction (similar to a unit circle)
                However, once theta > pi, it becomes negative
        """
        self.x = x
        self.y = y
        self.side_length = side_length
        self.subsection_cells = []

        self.delta_theta = float(2.0 * np.pi) / float(num_of_subsections)
        
        theta = 0.0
        # the node list will go in counter-clock wise direction
        for i in range(num_of_subsections):
            self.subsection_cells.append(self.Subsection_grid_cell_RRT(theta))
            theta = angle_wrap(theta + self.delta_theta)

    def has_node(self):
        """
        Return:
            True - if there is any RRT nodes in the the grid cell
        """
        for subsection in self.subsection_cells:
            if subsection.node_array != []:
                return True
        return False

    def __repr__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.subsection_cells)

    def __str__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.subsection_cells)

    class Subsection_grid_cell_RRT:
        def __init__(self, theta):
            self.theta = theta
            self.node_array = []

        def __repr__(self):
            return "Subsec: theta=" + str(self.theta) + ", node list: " + str(self.node_array)

        def __str__(self):
            return "Subsec: theta=" + str(self.theta) + ", node list: " + str(self.node_array)

"""
============================================================================

    Class - Planner RRT

============================================================================
"""
class Planner_RRT:
    """
    Class for RRT planning
    """
    def __init__(self, start, goal, boundary, obstacles, habitats, exp_rate = 1, dist_to_end = 2, diff_max = 0.5, freq = 50, cell_side_length = 2, subsections_in_cell = 4):
        '''
        Parameters:
            start - initial Motion_plan_state of AUV, [x, y, z, theta, v, w, time_stamp]
            goal - Motion_plan_state of the shark, [x, y, z]
            boundary - max & min Motion_plan_state of the configuration space [[x_min, y_min, z_min],[x_max, y_max, z_max]]
            obstacles - array of Motion_plan_state, representing obstacles [[x1, y1, z1, size1], [x2, y2, z2, size2] ...]
        '''
        # initialize start, goal, obstacle, boundaryfor path planning
        self.start = start
        self.goal = goal

        self.boundary_point = boundary
        self.cell_side_length = cell_side_length
        self.subsections_in_cell = subsections_in_cell

        # has node array
        # an 1D array determining whether a grid cell has any nodes
        # 0 = there isn't any nodes, 1 = there is at least 1 node
        self.has_node_array = []
        
        # a 1D array representing the number of nodes in each grid cell
        self.rrt_grid_1D_array_num_of_nodes_only = []

        # discretize the environment into grids
        self.discretize_env(self.cell_side_length, self.subsections_in_cell)
        
        self.occupied_grid_cells_array = []

        # add the start node to the grid
        self.add_node_to_grid(self.start)
        
        self.obstacle_list = obstacles
        # testing data for habitats
        self.habitats = habitats
        
        # a list of motion_plan_state
        self.mps_list = [self.start]

        # if minimum path length is not achieved within maximum iteration, return the latest path
        self.last_path = []

        # setting parameters for path planning
        self.exp_rate = exp_rate
        self.dist_to_end = dist_to_end
        self.diff_max = diff_max
        self.freq = freq

        self.t_start = time.time()


    def discretize_env(self, cell_side_length, subsections_in_cell):
        """
        Separate the environment into grid
        """
        env_btm_left_corner = self.boundary_point[0]
        env_top_right_corner = self.boundary_point[1]
        env_width = env_top_right_corner.x - env_btm_left_corner.x
        env_height = env_top_right_corner.y - env_btm_left_corner.y

        self.env_grid = []

        self.size_of_row = int(env_height) // int(cell_side_length)
        self.size_of_col = int(env_width) // int(cell_side_length)

        for row in range(self.size_of_row):
            self.env_grid.append([])
            for col in range(self.size_of_col):
                env_cell_x = env_btm_left_corner.x + col * cell_side_length
                env_cell_y = env_btm_left_corner.y + row * cell_side_length
                self.env_grid[row].append(Grid_cell_RRT(env_cell_x, env_cell_y, side_length = cell_side_length, num_of_subsections=subsections_in_cell))
                # initialize the has_node_array and rrt_grid_1D_array_num_of_nodes_only
                self.has_node_array += [0] * self.subsections_in_cell
                self.rrt_grid_1D_array_num_of_nodes_only += [0] * self.subsections_in_cell


    def print_env_grid(self):
        """
        Print the environment grid
            - Each grid cell will take up a line
            - Rows will be separated by ----
        """
        for row in self.env_grid:
            for grid_cell in row:
                print(grid_cell)
            print("----")

    
    def add_node_to_grid(self, mps, remove_cell_with_many_nodes = False):
        """

        Parameter:
            mps - a motion plan state object, represent the RRT node that we are trying add to the grid
        """
        # from the x and y position, we can figure out which grid cell does the new node belong to
        hab_index_row = int(mps.y / self.cell_side_length)
        hab_index_col = int(mps.x / self.cell_side_length)

        if hab_index_row >= len(self.env_grid):
            print("auv is out of the habitat environment bound verticaly")
            return
        
        if hab_index_col >= len(self.env_grid[0]):
            print("auv is out of the habitat environment bound horizontally")
            return

        # then, we need to figure out which subsection dooes the new node belong to
        raw_hab_index_subsection = mps.theta / self.env_grid[hab_index_row][hab_index_col].delta_theta

        # round down the index to an integer
        hab_index_subsection = math.floor(raw_hab_index_subsection)

        # if index >= 0, then it has already found the right subsection
        # however, if index < 0, we have to do an extra step to find the right index
        if hab_index_subsection < 0:
            hab_index_subsection = int(self.subsections_in_cell + hab_index_subsection)

        if hab_index_subsection == self.subsections_in_cell:
            print("why invalid subsection ;-;")
            print(mps)
            print("found roow and col")
            print(hab_index_row)
            print(hab_index_col)
            print("all cells")
            print(self.env_grid[hab_index_row][hab_index_col])
            print("subsections")
            print(self.env_grid[hab_index_row][hab_index_col].subsection_cells)
            print("raw")
            print(raw_hab_index_subsection)
            print("found index")
            print(hab_index_subsection)
            text = input("stop")

            hab_index_subsection -= 1

        self.env_grid[hab_index_row][hab_index_col].subsection_cells[hab_index_subsection].node_array.append(mps)
        
        index_in_1D_array = hab_index_row * self.size_of_col * self.subsections_in_cell + hab_index_col * self.subsections_in_cell + hab_index_subsection
        # increase the counter for the number of nodes in the grid cell
        self.rrt_grid_1D_array_num_of_nodes_only[index_in_1D_array] += 1

        # add the grid cell into the occupied grid cell array if it hasn't been added
        if len(self.env_grid[hab_index_row][hab_index_col].subsection_cells[hab_index_subsection].node_array) == 1:
            self.occupied_grid_cells_array.append((hab_index_row, hab_index_col, hab_index_subsection))

            self.has_node_array[index_in_1D_array] += 1

        if remove_cell_with_many_nodes and self.rrt_grid_1D_array_num_of_nodes_only[index_in_1D_array] > NODE_THRESHOLD:
            return True
        else:
            return False


    def planning(self, max_step = 200, min_length = 250, plan_time=True):
        """
        RRT path planning with a specific goal

        path planning will terminate when:
            1. the path has reached a goal
            2. the maximum planning time has passed

        Parameters:
            start -  an np array
            animation - flag for animation on or off

        """
        path = []

        # self.init_live_graph()

        step = 0

        start_time = time.time()
    
        for _ in range(max_step):

            # pick the row index and col index for the grid cell where the tree will get expanded
            grid_cell_row, grid_cell_col, grid_cell_subsection = random.choice(self.occupied_grid_cells_array)

            done, path = self.generate_one_node((grid_cell_row, grid_cell_col, grid_cell_subsection))

            # if (not done) and path != None:
            #     self.draw_graph(path)
            # elif done:
            #     plt.plot([mps.x for mps in path], [mps.y for mps in path], '-r')

            step += 1

            if done:
                break
        
        actual_time_duration = time.time() - start_time

        return path, step, actual_time_duration


    def generate_one_node(self, grid_cell_index, step_num = None, min_length=250, remove_cell_with_many_nodes = False):
        """
        Based on the grid cell, randomly pick a node to expand the tree from from

        Return:
            done - True if we have found a collision-free path from the start to the goal
            path - the collision-free path if there is one, otherwise it's null
            new_node
        """
        grid_cell_row, grid_cell_col, grid_cell_subsection = grid_cell_index

        grid_cell = self.env_grid[grid_cell_row][grid_cell_col].subsection_cells[grid_cell_subsection]

        if grid_cell.node_array == []:
            print("hmmmm invalid grid cell pick")     
            print(grid_cell)
            print("node list")
            print(grid_cell.node_array)
            text = input("stop")
            return False, None

        # randomly pick a node from the grid cell   
        rand_node = random.choice(grid_cell.node_array)

        new_node = self.steer(rand_node, self.dist_to_end, self.diff_max, self.freq, step_num=step_num)

        valid_new_node = False
        
        # only add the new node if it's collision free
        if self.check_collision_free(new_node, self.obstacle_list):
            new_node.parent = rand_node
            new_node.length += rand_node.length
            self.mps_list.append(new_node)
            valid_new_node = True

            remove_the_chosen_grid_cell = self.add_node_to_grid(new_node, remove_cell_with_many_nodes=remove_cell_with_many_nodes)

            if remove_cell_with_many_nodes and remove_the_chosen_grid_cell:
                index_in_1D_array = grid_cell_row * self.size_of_col * self.subsections_in_cell + grid_cell_col * self.subsections_in_cell + grid_cell_subsection
                print("too many nodes generated from this grid cell")
                self.has_node_array[index_in_1D_array] = 0

        final_node = self.connect_to_goal_curve_alt(self.mps_list[-1], self.exp_rate, step_num=step_num)

        # if we can create a path between the newly generated node and the goal
        if self.check_collision_free(final_node, self.obstacle_list):
            final_node.parent = self.mps_list[-1]
            path = self.generate_final_course(final_node)   
            return True, path
        
        if valid_new_node:
            return False, new_node
        else:
            return False, None


    def steer(self, mps, dist_to_end, diff_max, freq, velocity = 1, traj_time_stamp = False, step_num = None):
        """
        """
        if traj_time_stamp:
            new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, traj_time_stamp = mps.traj_time_stamp, rl_state_id = step_num)
        else:
            new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, plan_time_stamp = time.time()-self.t_start, traj_time_stamp = mps.traj_time_stamp, rl_state_id = step_num)

        new_mps.path = [mps]

        n_expand = random.uniform(0, freq)
        n_expand = math.floor(n_expand/1)
        for _ in range(n_expand):
            #setting random parameters
            dist = random.uniform(0, dist_to_end)  # setting random range
            diff = random.uniform(-diff_max, diff_max)  # setting random range

            if abs(dist) > abs(diff):
                s1 = dist + diff
                s2 = dist - diff
                radius = (s1 + s2)/(-s1 + s2)
                phi = (s1 + s2)/ (2 * radius)
                
                ori_theta = new_mps.theta
                new_mps.theta = self.angle_wrap(new_mps.theta + phi)
                delta_x = radius * (math.sin(new_mps.theta) - math.sin(ori_theta))
                delta_y = radius * (-math.cos(new_mps.theta) + math.cos(ori_theta))
                new_mps.x += delta_x
                new_mps.y += delta_y
                if traj_time_stamp:
                    new_mps.traj_time_stamp += (math.sqrt(delta_x ** 2 + delta_y ** 2)) / velocity
                else:
                    new_mps.plan_time_stamp = time.time() - self.t_start
                    new_mps.traj_time_stamp += (math.sqrt(delta_x ** 2 + delta_y ** 2)) / velocity
                new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta=new_mps.theta, traj_time_stamp=new_mps.traj_time_stamp, plan_time_stamp=new_mps.plan_time_stamp, rl_state_id = step_num))

        new_mps.path[0] = mps

        return new_mps


    def connect_to_goal(self, mps, exp_rate, dist_to_end=float("inf")):
        new_mps = Motion_plan_state(mps.x, mps.y)
        d, theta = self.get_distance_angle(new_mps, self.goal)

        new_mps.path = [new_mps]

        if dist_to_end > d:
            dist_to_end = d

        n_expand = math.floor(dist_to_end / exp_rate)

        for _ in range(n_expand):
            new_mps.x += exp_rate * math.cos(theta)
            new_mps.y += exp_rate * math.sin(theta)
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y))

        d, _ = self.get_distance_angle(new_mps, self.goal)
        if d <= dist_to_end:
            new_mps.path.append(self.goal)
        
        new_mps.path[0] = mps

        return new_mps
    

    def generate_final_course(self, mps):
        path = [mps]
        mps = mps
        while mps.parent is not None:
            reversed_path = reversed(mps.path)
            for point in reversed_path:
                path.append(point)
            mps = mps.parent
        #path.append(mps)

        return path

    
    def init_live_graph(self):
        _, self.ax = plt.subplots()

        for row in self.env_grid:
            for grid_cell in row:
                cell = Rectangle((grid_cell.x, grid_cell.y), width=grid_cell.side_length, height=grid_cell.side_length, color='#2a753e', fill=False)
                self.ax.add_patch(cell)
        
        for obstacle in self.obstacle_list:
            self.plot_circle(obstacle.x, obstacle.y, obstacle.size)

        self.ax.plot(self.start.x, self.start.y, "xr")
        self.ax.plot(self.goal.x, self.goal.y, "xr")

    
    def draw_graph(self, rnd=None):
        # plt.clf()  # if we want to clear the plot
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd != None:
            # self.ax.plot(rnd.x, rnd.y, ",", color="#000000")

            plt.plot([point.x for point in rnd.path], [point.y for point in rnd.path], '-', color="#000000")
            plt.plot(rnd.x, rnd.y, 'o', color="#000000")
        else:
            for mps in self.mps_list:
                if mps.parent:
                    plt.plot([point.x for point in mps.path], [point.y for point in mps.path], '-g')

        plt.axis("equal")

        plt.pause(1)


    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)
    
    
    def connect_to_goal_curve_alt(self, mps, exp_rate, step_num = None):
        new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, traj_time_stamp = mps.traj_time_stamp, rl_state_id = step_num)
        theta_0 = new_mps.theta
        _, theta = self.get_distance_angle(mps, self.goal)
        diff = theta - theta_0
        diff = self.angle_wrap(diff)
        
        if abs(diff) > math.pi / 2:
            return

        #polar coordinate
        r_G = math.hypot(self.goal.x - new_mps.x, self.goal.y - new_mps.y)
        phi_G = math.atan2(self.goal.y - new_mps.y, self.goal.x - new_mps.x)


        # arc
        if phi_G - new_mps.theta != 0:
            phi = 2 * self.angle_wrap(phi_G - new_mps.theta)
            # prevent a dividing by 0 error
        else:
            return
        
        if math.sin(phi_G - new_mps.theta) != 0:
            radius = r_G / (2 * math.sin(phi_G - new_mps.theta))
        else:
            return

        length = radius * phi
        if phi > math.pi:
            phi -= 2 * math.pi
            length = -radius * phi
        elif phi < -math.pi:
            phi += 2 * math.pi
            length = -radius * phi
        new_mps.length += length

        ang_vel = phi / (length / exp_rate)

        #center of rotation
        x_C = new_mps.x - radius * math.sin(new_mps.theta)
        y_C = new_mps.y + radius * math.cos(new_mps.theta)

        n_expand = math.floor(length / exp_rate)
        for i in range(n_expand+1):
            new_mps.x = x_C + radius * math.sin(ang_vel * i + theta_0)
            new_mps.y = y_C - radius * math.cos(ang_vel * i + theta_0)
            new_mps.theta = ang_vel * i + theta_0
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta = new_mps.theta, plan_time_stamp=time.time()-self.t_start, rl_state_id = step_num))
        
        return new_mps

    def angle_wrap(self, ang):
        if -math.pi <= ang <= math.pi:
            return ang
        elif ang > math.pi: 
            ang += (-2 * math.pi)
            return self.angle_wrap(ang)
        elif ang < -math.pi: 
            ang += (2 * math.pi)
            return self.angle_wrap(ang)

    def check_collision_free(self, mps, obstacleList):
        """
        Collision
        Return:
            True -  if the new node (as a motion plan state) and its path is collision free
            False - otherwise
        """
        if mps is None:
            return False

        dList = []
        for obstacle in obstacleList:
            for point in mps.path:
               d, _ = self.get_distance_angle(obstacle, point)
               dList.append(d) 

            if min(dList) <= obstacle.size:
                return False  # collision
    
        for point in mps.path:
            if not self.check_within_boundary(point):
                return False

        return True  # safe
    

    def check_collision_obstacle(self, mps, obstacleList):
        for obstacle in obstacleList:
            d, _ = self.get_distance_angle(obstacle, mps)
            if d <= obstacle.size:
                return False
        return True


    def check_within_boundary(self, mps):
        """
        Warning: 
            For a rectangular environment only

        Return:
            True - if it's within the environment boundary
            False - otherwise
        """
        env_btm_left_corner = self.boundary_point[0]
        env_top_right_corner = self.boundary_point[1]

        within_x_bound = (mps.x >= env_btm_left_corner.x) and (mps.x <= env_top_right_corner.x)
        within_y_bound = (mps.y >= env_btm_left_corner.y) and (mps.y <= env_top_right_corner.y)

        return (within_x_bound and within_y_bound)

    def get_distance_angle(self, start_mps, end_mps):
        """
        Return
            - the range and 
            - the bearing between 2 points, represented as 2 Motion_plan_states
        """
        dx = end_mps.x-start_mps.x
        dy = end_mps.y-start_mps.y
        #dz = end_mps.z-start_mps.z
        dist = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy,dx)
        return dist, theta
    

    def cal_length(self, path):
        length = 0
        for i in range(1, len(path)):
            length += math.sqrt((path[i].x-path[i-1].x)**2 + (path[i].y-path[i-1].y)**2)
        return length


"""
============================================================================

    Class - RRT Env

============================================================================
"""
class RRTEnv(gym.Env):
    # possible render modes: human, rgb_array (creates image for making videos), ansi (string)
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Declare the data members without initialize them
            automatically called when we build an environment with gym.make('gym_auv:auv-v0')

        Warning: 
            Need to immediately call init_env function to actually initialize the environment
        """
        # action: 
        #   a tuple of (v, w), linear velocity and angular velocity
        # range for v (unit: m/s): [-AUV_MAX_V, AUV_MAX_V]
        # range for w (unit: radians): [-AUV_MAX_W, AUV_MAX_W]
        self.action_space = spaces.Box(low = np.array([-AUV_MAX_W]), high = np.array([AUV_MAX_W]), dtype = np.float64)

        self.observation_space = None

        self.auv_init_pos = None
        self.shark_init_pos = None

        # the current state that the env is in
        self.state = None

        self.obstacle_array = []
        self.obstacle_array_for_rendering = []

        self.habitats_array = []
        self.habitats_array_for_rendering = []

        # self.live_graph = Live3DGraph(PLOT_3D)

        self.auv_x_array_plot = []
        self.auv_y_array_plot = []
        self.auv_z_array_plot = []

        self.shark_x_array_plot = []
        self.shark_y_array_plot = []
        self.shark_z_array_plot = []

        self.visited_unique_habitat_count = 0


    def init_env(self, auv_init_pos, shark_init_pos, boundary_array, grid_cell_side_length, num_of_subsections, obstacle_array = [], habitat_grid = None):
        """
        Initialize the environment based on the auv and shark's initial position

        Parameters:
            auv_init_pos - an motion plan state object
            shark_init_pos - an motion plan state object
            boundary_array - an array of 2 motion plan state objects
                TODO: For now, let the environment be a rectangle
                1st mps represents the bottom left corner of the env
                2nd mps represents the upper right corner of the env
            obstacle_array - an array of motion plan state objects
            habitat_grid - an HabitatGrid object (discretize the environment into grid)

        Return:
            a dictionary with the initial observation of the environment
        """
        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos

        self.obstacle_array_for_rendering = obstacle_array
        
        self.habitats_array_for_rendering = []
        if habitat_grid != None:
            self.habitat_grid = habitat_grid
            self.habitats_array_for_rendering = habitat_grid.habitat_array

        self.obstacle_array = []
        for obs in obstacle_array:
            self.obstacle_array.append([obs.x, obs.y, obs.z, obs.size])
        self.obstacle_array = np.array(self.obstacle_array)

        self.boundary_array = boundary_array

        self.cell_side_length = grid_cell_side_length
        self.num_of_subsections = num_of_subsections

        # declare the observation space (required by OpenAI)
        self.observation_space = spaces.Dict({
            'auv_pos': spaces.Box(low = np.array([auv_init_pos.x - ENV_SIZE, auv_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([auv_init_pos.x + ENV_SIZE, auv_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'shark_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'obstacles_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'habitats_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64)
        })

        # initialize the 
        self.init_data_for_plot(auv_init_pos, shark_init_pos)
        
        return self.reset()

    def step(self, chosen_grid_cell_idx, step_num):
        """
        In each step, we will generate an additional node in the RRT tree.

        Parameter:
            action - a tuple, representing the linear velocity and angular velocity of the auv

        Return:
            observation - a tuple of 2 np array, representing the auv and shark's new position
                each array has the format: [x_pos, y_pos, z_pos, theta]
            reward - float, amount of reward returned after previous action
            done - float, whether the episode has ended
            info - dictionary, can provide debugging info (TODO: right now, it's just an empty one)
        """
        # print("picked unprocessed index")
        # print(chosen_grid_cell_idx)

        # print("state")
        # for i in range(len(self.state["rrt_grid"])):
        #     print(str(i) + " : " + str(self.state["rrt_grid"][i]))

        grid_cell_index = chosen_grid_cell_idx // self.num_of_subsections
        subsection_index = chosen_grid_cell_idx % self.num_of_subsections
        
        # print("grid cell index")
        # print(grid_cell_index)

        # print("subsection index")
        # print(subsection_index)

        # convert the index for grid cells in the 1D array back to 2D array
        chosen_grid_cell_row_idx = grid_cell_index // len(self.rrt_planner.env_grid[0])
        chosen_grid_cell_col_idx = grid_cell_index % len(self.rrt_planner.env_grid[0])

        # print("row and col")
        # print(chosen_grid_cell_row_idx)
        # print(chosen_grid_cell_col_idx)

        # text = input("stop")

        # chosen_grid_cell = self.rrt_planner.env_grid[chosen_grid_cell_row_idx][chosen_grid_cell_col_idx].subsection_cells[subsection_index]

        done, path = self.rrt_planner.generate_one_node((chosen_grid_cell_row_idx, chosen_grid_cell_col_idx, subsection_index), step_num, remove_cell_with_many_nodes=REMOVE_CELL_WITH_MANY_NODES)

        # TODO: how we are updating the grid's info and the has node array is very inefficient

        self.state["has_node"] = np.array(self.rrt_planner.has_node_array)

        self.state["rrt_grid_num_of_nodes_only"] = np.array(self.rrt_planner.rrt_grid_1D_array_num_of_nodes_only)

        if path != None:
            self.state["path"] = path

        # if the RRT planner has found a path in this step
        if done and path != None:
            reward = R_FOUND_PATH
        elif path != None:
            # if the RRT planner adds a new node
            reward = R_CREATE_NODE
        else:
            # TODO: For now, the reward encourages using less time to plan the path
            reward = R_INVALID_NODE
        
        
        return self.state, reward, done, {}


    def convert_rrt_grid_to_1D (self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """
        rrt_grid_1D_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    rrt_grid_1D_array.append([grid_cell.x, grid_cell.y, subsection.theta, len(subsection.node_array)])

        return np.array(rrt_grid_1D_array)


    def convert_rrt_grid_to_1D_num_of_nodes_only(self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """
        rrt_grid_1D_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    rrt_grid_1D_array.append(len(subsection.node_array))

        return np.array(rrt_grid_1D_array)

    def generate_rrt_grid_has_node_array (self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """

        has_node_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    if len(subsection.node_array) == 0:
                        has_node_array.append(0)
                    else:
                        has_node_array.append(1)

        return np.array(has_node_array)

    
    def calculate_range(self, a_pos, b_pos):
        """
        Calculate the range (distance) between point a and b, specified by their coordinates

        Parameters:
            a_pos - an array / a numpy array
            b_pos - an array / a numpy array
                both have the format: [x_pos, y_pos, z_pos, theta]

        TODO: include z pos in future range calculation?
        """
        a_x = a_pos[0]
        a_y = a_pos[1]
        b_x = b_pos[0]
        b_y = b_pos[1]

        delta_x = b_x - a_x
        delta_y = b_y - a_y

        return np.sqrt(delta_x**2 + delta_y**2)

    
    def within_follow_range(self, auv_pos, shark_pos):
        """
        Check if the auv is within FOLLOWING_RADIUS of the shark

        Parameters:
            auv_pos - an array / a numpy array
            shark_pos - an array / a numpy array
                both have the format: [x_pos, y_pos, z_pos, theta]
        """
        auv_shark_range = self.calculate_range(auv_pos, shark_pos)
        if auv_shark_range <= FOLLOWING_RADIUS:
            if DEBUG:
                print("Within the following range")
            return True
        else:
            return False

    
    def check_collision(self, auv_pos):
        """
        Check if the auv at the current state is hitting any obstacles

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= obs[3]:
                print("Hit an obstacle")
                return True
        return False


    def check_close_to_obstacles(self, auv_pos):
        """
        Check if the auv at the current state is close to any obstacles
        (Within a circular region with radius: obstacle's radius + OBSTACLE_ZONE)

        Parameter:
            auv_pos - an array / a np array [x, y, z, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= (obs[3] + OBSTACLE_ZONE):
                if DEBUG: 
                    print("Close to an obstacles")
                return True
        return False


    def check_close_to_walls(self, auv_pos, dist_from_walls_array):
        for dist_from_wall in dist_from_walls_array:
            if dist_from_wall <= WALL_ZONE:
                if DEBUG:
                    print("Close to the wall")
                return True
        return False
    

    def update_num_time_visited_for_habitats(self, habitats_array, visited_habitat_cell):
        """
        Update the number of times visited for a habitat that has been visited by the auv in the current timestep

        Parameters:
            habitats_array - an array of arrays, where each array represent the state of a habitat
                format: [[hab1_x, hab1_y, hab1_side_length, hab1_num_time_visited], [hab2_x, hab2_y, hab2_side_length, hab2_num_time_visited], ...]
                Warning: this function does not modify habitats_array
            visited_habitat_cell - a HabitatCell object, indicating the current habitat cell that the auv is in
                its id indicates which habitat's num_time_visited should be incremented by 1

        Return:
            a new copy of the habitats_array with the updated number of time visited 
        """
        # print(visited_habitat_cell)

        # make a deep copy of the original habitats array to ensure that we are not modifying the habitats_array that gets passed in
        new_habitats_array = copy.deepcopy(habitats_array)

        # double check that the auv has actually visited a habitat cell and is not outside of the habitat grid
        if visited_habitat_cell != False:
            habitat_index = visited_habitat_cell.habitat_id

            # the 3rd element represents the number of times the AUV has visited an habitat
            new_habitats_array[habitat_index][3] += 1
        
        return new_habitats_array
    

    def reset(self):
        """
        Reset the environment
            - Set the observation to the initial auv and shark position
            - Reset the habitat data (the number of times the auv has visited them)

        Return:
            a dictionary with the initial observation of the environment
        """
        # reset the habitat array, make sure that the number of time visited is cleared to 0
        # self.habitats_array = []
        # for hab in self.habitats_array_for_rendering:
        #     self.habitats_array.append([hab.x, hab.y, hab.side_length, hab.num_of_time_visited])
        # self.habitats_array = np.array(self.habitats_array)

        # reset the count for how many unique habitat had the auv visited
        self.visited_unique_habitat_count = 0
        # reset the count for how many time steps had the auv visited an habitat
        self.total_time_in_hab = 0

        auv_init_pos = np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta])

        shark_init_pos = np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z, self.shark_init_pos.theta])

        # initialize the RRT planner
        self.rrt_planner = Planner_RRT(self.auv_init_pos, self.shark_init_pos, self.boundary_array, self.obstacle_array_for_rendering, self.habitats_array_for_rendering, cell_side_length = self.cell_side_length, freq=RRT_PLANNER_FREQ, subsections_in_cell = self.num_of_subsections)

        # rrt_grid_1D_array = self.convert_rrt_grid_to_1D(self.rrt_planner.env_grid)
        # rrt_grid_1D_array_num_of_nodes_only = self.convert_rrt_grid_to_1D_num_of_nodes_only(self.rrt_planner.env_grid)
        # has_node_array = self.generate_rrt_grid_has_node_array(self.rrt_planner.env_grid)

        self.state = {
            'auv_pos': auv_init_pos,\
            'shark_pos': shark_init_pos,\
            'obstacles_pos': self.obstacle_array,\
            'has_node': np.array(self.rrt_planner.has_node_array),\
            'path': None,\
            'rrt_grid_num_of_nodes_only': np.array(self.rrt_planner.rrt_grid_1D_array_num_of_nodes_only),\
        }

        # print("initial state")
        # print(has_node_array)
        # print(self.state["has_node"])
        # print(has_node_array == self.state["has_node"])
        # print("---")
        # print(rrt_grid_1D_array_num_of_nodes_only)
        # print(self.state["rrt_grid_num_of_nodes_only"])
        # print(rrt_grid_1D_array_num_of_nodes_only == self.state["rrt_grid_num_of_nodes_only"])
        # text = input("stop")

        return self.state


    def render(self, mode='human', print_state = True):
        """
        Render the environment by
            - printing out auv and shark's current position
            - returning self.state , so that another helper function can use plot the environment

        Return:
            a dictionary representing the current auv position, shark position, obstacles data, habitats data
        """
        
        """auv_pos = self.state['auv_pos']
        shark_pos = self.state['shark_pos']
        
        if print_state: 
            print("==========================")
            print("auv position: ")
            print("x = ", auv_pos[0], " y = ", auv_pos[1], " z = ", auv_pos[2], " theta = ", auv_pos[3])
            print("shark position: ")
            print("x = ", shark_pos[0], " y = ", shark_pos[1], " z = ", shark_pos[2], " theta = ", shark_pos[3])
            print("==========================")"""

        return self.state


    def render_2D_plot(self, new_state):
        """
        Render the environment in a 2D environment

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
            shark_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
        """
        if new_state != None and type(new_state) != list:
            # draw the new edge, which is not a successful path
            self.live_graph.ax_2D.plot([point.x for point in new_state.path], [point.y for point in new_state.path], '-', color="#000000")
            self.live_graph.ax_2D.plot(new_state.x, new_state.y, 'o', color="#000000")
        elif new_state != None and type(new_state) == list:
            # if we are supposed to draw the final path  
            # new_state is now a list of nodes
            self.live_graph.ax_2D.plot([node.x for node in new_state], [node.y for node in new_state], '-r')
            # self.ax.plot(rnd.x, rnd.y, ",", color="#000000")

        # pause so the plot can be updated
        plt.pause(0.0001)


    def init_live_graph(self, live_graph_2D):
        if live_graph_2D:
            self.live_graph.ax_2D.plot(self.auv_init_pos.x, self.auv_init_pos.y, "xr")
            self.live_graph.ax_2D.plot(self.shark_init_pos.x, self.shark_init_pos.y, "xr")

            if self.obstacle_array_for_rendering != []:
                self.live_graph.plot_obstacles_2D(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

            for row in self.rrt_planner.env_grid:
                for grid_cell in row:
                    cell = Rectangle((grid_cell.x, grid_cell.y), width=grid_cell.side_length, height=grid_cell.side_length, color='#2a753e', fill=False)
                    self.live_graph.ax_2D.add_patch(cell)
            
            self.live_graph.ax_2D.set_xlabel('X')
            self.live_graph.ax_2D.set_ylabel('Y')


    def init_data_for_plot(self, auv_init_pos, shark_init_pos):
        """
        """
        self.auv_x_array_plot = [auv_init_pos.x]
        self.auv_y_array_plot = [auv_init_pos.y]
        self.auv_z_array_plot = [auv_init_pos.z]

        self.shark_x_array_plot = [shark_init_pos.x]
        self.shark_y_array_plot = [shark_init_pos.y]
        self.shark_z_array_plot = [shark_init_pos.z]
