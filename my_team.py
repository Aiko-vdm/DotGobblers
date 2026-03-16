# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from util import Queue
#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class MiniMaxAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # TODO best be initialisatie?
        self.depth = 2

    def choose_action(self, game_state):
        """
        Returns the minimax action
        """
        # Chooses the best action by recursively searching for the best state_value that a legal action can provide
        # initial values:
        if hasattr(self, 'pos_history') and len(self.pos_history) >= 4:
            if (self.pos_history[-1] == self.pos_history[-3] and
                    self.pos_history[-2] == self.pos_history[-4]):
                legal_actions = game_state.get_legal_actions(self.index)
                current_dir = game_state.get_agent_state(self.index).configuration.direction
                non_reverse = [a for a in legal_actions
                               if a != Directions.REVERSE[current_dir]
                               and a != Directions.STOP]
                if non_reverse:
                    import random
                    return random.choice(non_reverse)

        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')  # initial best value for max so far on path to root
        beta = float('inf')  # initial best value for min so far on path to root
        legal_actions = [action for action in game_state.get_legal_actions(self.index) if
                         action != Directions.STOP]  # Get legal actions

        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            next_agent = (self.index + 1) % game_state.get_num_agents()
            state_value = self.__value(successor, next_agent, depth=self.depth, alpha=alpha,
                                       beta=beta)  # Start with the first ghost
            # update the best value with its associated action when a better value is found
            if state_value > best_value:
                best_value = state_value
                best_action = action

            # state value is outside of bounds, so the other actions (node children) should not be visited
            if state_value > beta:
                return best_action
            alpha = max(alpha, best_value)

        return best_action

    def __value(self, state, agent_index: int, depth: int, alpha, beta):
        """
        Dispatcher function that retrieves the achievable state value until a terminal state or depth limit is reached
        Deploys __max_value when our team's turn, and __min_value when opponent's turn.
        Method is private and should only be called in the main loop from the get_action method
        """
        if depth == 0 or state.is_over():
            return self.evaluate(state)
        agent_state = state.get_agent_state(agent_index)
        if agent_state.configuration is None:
            next_agent = (agent_index + 1) % state.get_num_agents()
            next_depth = depth - 1 if agent_index == state.get_num_agents() - 1 else depth
            return self.__value(state, next_agent, next_depth, alpha, beta)
        # our team's turn, so use max
        if agent_index == self.index:
            return self.__max_value(state, agent_index, depth, alpha, beta)
        elif agent_index in self.get_team(state):
            return self.__max_value(state, agent_index, depth, alpha, beta)
        # opponent's turn, so use min. Logic to decrement depth is implemented in the min function
        else:
            return self.__min_value(state, agent_index, depth, alpha, beta)

    def __max_value(self, state, agent_index: int, depth: int, alpha, beta):
        """
        Retrieves the max state value until a terminal state or reached depth limit
        Method is private and should only be called in the main loop from the __value method
        """
        state_value = float('-inf')
        legal_actions = [action for action in state.get_legal_actions(agent_index) if action != Directions.STOP]
        number_of_agents = state.get_num_agents()

        for action in legal_actions:
            successor = state.generate_successor(agent_index, action)
            next_agent_index = (agent_index + 1) % number_of_agents  # modulo to cycle through the agent indexes
            next_depth = depth - 1 if agent_index == number_of_agents - 1 else depth
            state_value = max(state_value, self.__value(successor, next_agent_index, next_depth, alpha, beta))

            # state value is outside of bounds, so the other actions (node children) should not be visited
            if state_value > beta:
                return state_value
            alpha = max(alpha, state_value)
        return state_value

    def __min_value(self, state, agent_index: int, depth: int, alpha, beta):
        """
        Retrieves the achievable min state value until a terminal state or depth limit is reached
        Method is private and should only be called in the main loop from the __value method
        """
        state_value = float('inf')
        legal_actions = [action for action in state.get_legal_actions(agent_index) if action != Directions.STOP]
        number_of_agents = state.get_num_agents()

        for action in legal_actions:
            successor = state.generate_successor(agent_index, action)
            next_agent_index = (agent_index + 1) % number_of_agents  # modulo to cycle through the agent indexes
            next_depth = depth - 1 if agent_index == number_of_agents - 1 else depth  # stop decrementing the depth at the last min opponent.
            state_value = min(state_value, self.__value(successor, next_agent_index, next_depth, alpha, beta))

            # state value is outside of bounds, so the other actions (node children) should not be visited
            if state_value < alpha:
                return state_value
            beta = min(beta, state_value)

        return state_value

    def evaluate(self, game_state):
        raise NotImplementedError


class MinimaxOffensiveAgent(MiniMaxAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.walls = game_state.get_walls()
        # FIXME: kan in principe statisch worden geïnitialiseerd bij init
        self.dead_ends = {}
        self.compute_dead_ends()
        # FIXME: beiden kunnen in principe statisch worden geïnitialiseerd bij init
        self.pos_history = []
        self.pos_hist_len = 4
        self.hesitation_ctr = 0
        self.avoided_cluster = None
        self.avoided_cluster_timer = 0

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        state = game_state.get_agent_state(self.index)
        if my_pos is not None:
            self.pos_history.append(my_pos)
            if len(self.pos_history) > self.pos_hist_len: self.pos_history.pop(0)
            if self.avoided_cluster_timer > 0: 
                self.avoided_cluster_timer -= 1 
            else: self.avoided_cluster = None
            if not state.is_pacman:
                enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
                if defenders:
                    min_dist = min(self.get_maze_distance(my_pos, a.get_position()) for a in defenders)
                    if min_dist <= 5:
                        self.hesitation_ctr += 1
                    else: self.hesitation_ctr = 0
                else: self.hesitation_ctr = 0
                if self.hesitation_ctr >= 5:
                    self.avoided_cluster = self.curr_target_cluster
                    self.avoided_cluster_timer = 20
                    self.hesitation_ctr = 0
            else: self.hesitation_counter = 0
        return super().choose_action(game_state)

    # TODO: make private/internal
    def compute_dead_ends(self):
        # FIXME: move import
        from util import Queue
        walls = self.walls
        neighbours = {}
        degree = {}

        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                not_wall = (x, y)
                list_of_neighbours = []
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    newx, newy = x + dx, y + dy

                    if not walls[newx][newy]:
                        list_of_neighbours.append((newx, newy))

                neighbours[not_wall] = list_of_neighbours
                degree[not_wall] = len(list_of_neighbours)

        queue = Queue()
        for not_wall in degree:
            if degree[not_wall] == 1:
                queue.push(not_wall)
                self.dead_ends[not_wall] = 1
                self.debug_draw(not_wall, color=(1, 1, 1))

        while not queue.is_empty():
            not_wall = queue.pop()
            for x in neighbours[not_wall]:
                if x not in degree: continue
                degree[x] -= 1

                if degree[x] == 1 and x not in self.dead_ends:
                    self.dead_ends[x] = self.dead_ends[not_wall] + 1
                    queue.push(x)
                    self.debug_draw(x, color=(1, 1, 1))

    def evaluate(self, game_state):
        features = self.get_features(game_state)
        weights = self.get_weights()

        return features * weights

    def get_features(self, game_state):
        features = util.Counter()
        food_list = self.get_food(game_state).as_list()

        radius = 2  # beste radius??

        # ------- compute_clusters START
        clusters = []
        for food in food_list:
            count = 0
            for rest_food in food_list:
                if self.get_maze_distance(food, rest_food) <= radius:
                    count += 1
            clusters.append((food, count))
        # -------- compute_clusters END
        best_food = None
        best_cluster_size = 0

        for food, size in clusters:
            if food == self.avoided_cluster: continue
            if size > best_cluster_size:
                best_cluster_size = size
                best_food = food

        if best_food is None:
            for food, size in clusters:
                if size > best_cluster_size:
                    best_cluster_size = size
                    best_food = food

        self.curr_target_cluster = best_food
        state = game_state.get_agent_state(self.index)
        my_pos = state.get_position()

        # TODO refactor naar lijn hierboven? my_pos = state.get_position() if state.get_position() is not none else 0
        if my_pos is None:
            return 0

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        active_defenders = [a for a in defenders if a.scared_timer == 0]
        scared_defenders = [a for a in defenders if a.scared_timer > 0]
        is_chased = False
        closest_defender_dist = float('inf')
        if active_defenders:
            defender_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in active_defenders]
            closest_defender_dist = min(defender_dists)
            is_chased = closest_defender_dist <= 5

        if state.is_pacman and closest_defender_dist <= 5:
            features['ghost_proximity'] = 10 - closest_defender_dist

        features['score'] = self.get_score(game_state)

        features['uneaten_food'] = len(food_list)  # self.get_score(successor)

        if best_food is not None:
            distance = self.get_maze_distance(my_pos, best_food)
            features['distance_to_cluster'] = distance
            features['cluster_size'] = best_cluster_size

        if scared_defenders:
            min_scared_timer = min(a.scared_timer for a in scared_defenders)
            if min_scared_timer >= 5:
                features['return_home'] = 0
                features['dead_end'] = 0
                scared_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in scared_defenders]
                features['dist_to_scared_defender'] = min(scared_dists)
        else:
            carrying = state.num_carrying
            distance_to_home = self.get_maze_distance(my_pos, self.start)
            time_left = game_state.data.timeleft
            time = 1200
            urgency = 1 - time_left / time
            if is_chased:
                capsules = self.get_capsules(game_state)
                if capsules:
                    capsule_dists = [self.get_maze_distance(my_pos, capsule) for capsule in capsules]
                    features['dist_to_capsule'] = min(capsule_dists)
                else:
                    features['return_home'] = carrying * distance_to_home * urgency * 5
            else:
                features['return_home'] = carrying * distance_to_home * urgency

        if active_defenders:
            for defender in active_defenders:
                defender_pos = defender.get_position()
                if my_pos == defender_pos: features['walk_into_defender'] = 1
        if my_pos in self.dead_ends:
            depth = self.dead_ends[my_pos]
            if closest_defender_dist <= depth * 2:
                features['dead_end'] = 1

        if self.pos_history:
            count = self.pos_history.count(my_pos)
            features['reverse'] = count

        return features

    def get_weights(self):
        weights = {'score': 100,
                   'uneaten_food': -100,
                   'distance_to_cluster': -1,
                   'cluster_size': 1,
                   'return_home': -1,
                   'dead_end': -75,
                   'reverse': 0,
                   'ghost_proximity': -10,
                   'dist_to_capsule': -10,
                   'walk_into_defender': -100,
                   'dist_to_scared_defender': -15}
        return weights


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.dead_ends = {}
        CaptureAgent.register_initial_state(self, game_state)
        self.compute_dead_ends()
        

    def compute_dead_ends(self):
        # FIXME: move import
        walls = self.walls
        neighbours = {}
        degree = {}

        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                not_wall = (x, y)
                list_of_neighbours = []
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    newx, newy = x + dx, y + dy

                    if not walls[newx][newy]:
                        list_of_neighbours.append((newx, newy))

                neighbours[not_wall] = list_of_neighbours
                degree[not_wall] = len(list_of_neighbours)

        queue = Queue()
        for not_wall in degree:
            if degree[not_wall] == 1:
                queue.push(not_wall)
                self.dead_ends[not_wall] = 1
                self.debug_draw(not_wall, color=(1, 1, 1))

        while not queue.is_empty():
            not_wall = queue.pop()
            for x in neighbours[not_wall]:
                if x not in degree: continue
                degree[x] -= 1

                if degree[x] == 1 and x not in self.dead_ends:
                    self.dead_ends[x] = self.dead_ends[not_wall] + 1
                    queue.push(x)
                    self.debug_draw(x, color=(1, 1, 1))

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.bottleneck_positions = None
        self.high_traffic_positions = None


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
        self.last_eaten_food = None
        #TODO: see if still useful
        #self.find_high_traffic(game_state)
        self.find_bottlenecks(game_state)



        def draw_bottlenecks():
            for bottleneck in self.bottleneck_positions:
                self.debug_draw(bottleneck, color=(158,224,32))
        # easier to comment out in one line
        draw_bottlenecks()

    def find_bottlenecks(self, game_state):
        def pos_is_gate(row,col,game_state):
            if all([not game_state.has_wall(col, row),
                    not game_state.has_wall(col + 1, row),
                    not game_state.has_wall(col - 1, row),
                    game_state.has_wall(col, row - 1),
                    game_state.has_wall(col, row + 1)]):
                return True
            else:
                return False

        middle_x = (game_state.data.layout.width - 1) // 2 if self.red else game_state.data.layout.width // 2
        defense_midfield_x = middle_x - middle_x // 2 if self.red else middle_x + middle_x // 2
        result = []
        maze_height = game_state.data.layout.height
        for col in range(middle_x,defense_midfield_x):
            for row in range(1,maze_height - 1 ):
                if pos_is_gate(row,col,game_state):
                   # self.debug_draw((col,row), (122,244,32))
                    result.append((col, row))
        self.bottleneck_positions = result

    def find_high_traffic(self, game_state):
        agenda = util.Queue()
        closed = util.Counter()
        agenda.push(game_state)  # we push the starting state
        middle_x = (game_state.data.layout.width - 1) // 2 if self.red else game_state.data.layout.width // 2
        defense_midfield_x = middle_x - middle_x // 2 if self.red else middle_x + middle_x // 2

        midfield_pos = [(defense_midfield_x,y) for y in range(1,17) if not game_state.has_wall(defense_midfield_x, y)]

        end_positions = [(middle_x, y) for y in range(1, 17) if not game_state.has_wall(middle_x,y)]
        while not agenda.is_empty():
            current_state = agenda.pop()
            # if current_state.get_agent_position(self.index) == end_position:
            if current_state.get_agent_position(self.index) in end_positions:
                high_trafic_points = []
                for i in range(0, 10):
                    pos = closed.arg_max()
                    closed[pos] = 0
                    high_trafic_points.append(pos)
                #self.bottlenecks = bottlenecks
                self.high_traffic_positions = closed.sorted_keys()[:5]

            elif current_state not in closed:
                closed[current_state.get_agent_position(self.index)] = 1

                legal_actions = current_state.get_legal_actions(self.index)
                successor_states = [current_state.generate_successor(self.index, action) for action in legal_actions]
                for successor_state in successor_states:
                    if successor_state.get_agent_position(self.index) not in closed:
                        agenda.push(successor_state)
                    else:
                        visited_position = successor_state.get_agent_position(self.index)
                        closed[visited_position] += 1
        #draw end-goal boundry
        # for pos in end_positions:
        #     self.debug_draw(pos, (122,244,32))
        # for pos in midfield_pos:
        #     self.debug_draw(pos, (122,244,32))
    
    def get_features(self, game_state, action):
        features = util.Counter()
        current_food = self.get_food_you_are_defending(game_state).as_list()

        # code bellow checks if our food is eaten and returns the closest position for which this is the case
        eaten = set(self.previous_food) - set(current_food)
        if eaten:
            pos = game_state.get_agent_state(self.index).get_position()
            min_dist = float('inf')
            closest = None
            for food in eaten:
                dist = self.get_maze_distance(pos, food)
                if dist < min_dist:
                    min_dist = dist
                    closest = food
            #FIXME: in init?
            self.last_eaten_food = closest
        #FIXME: in init?
        self.previous_food = current_food

        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        # TODO: check if still useful
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            # Of those we see, how many are trapped in dead ends
            dist_trapped = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies if a in self.dead_ends]
            features['trapped_invader_distance'] = min(dist_trapped) if len(dist_trapped) > 0 else 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(invaders) == 0 and self.last_eaten_food is not None:
            dist = self.get_maze_distance(my_pos, self.last_eaten_food)
            features['distance_to_last_eaten_food'] = dist

        #distance to a bottleneck
        bottleneck_dist = [self.get_maze_distance(my_pos, bottleneck) for bottleneck in self.bottleneck_positions]
        features['bottleneck_distance'] = min(bottleneck_dist)

        return features

    def get_weights(self, game_state, action):
        def is_scared():
            if game_state.get_agent_state(self.index).scared_timer > 0:
                return True
            else:
                return False
        invader_distance_w = -10 if not is_scared() else 5
        trapped_invader_distance_w = -100 if not is_scared() else 50
        return {'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': invader_distance_w,
                'trapped_invader_distance': trapped_invader_distance_w,
                'stop': -100,
                'reverse': -2,
                'distance_to_last_eaten_food': -10,
                'bottleneck_distance': -5}
    
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.pos_history = []
        self.pos_hist_len = 4

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        state = game_state.get_agent_state(self.index)

        if my_pos is not None:
            self.pos_history.append(my_pos)
            if len(self.pos_history) > self.pos_hist_len:
                self.pos_history.pop(0)
        return super().choose_action(game_state)
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        radius = 2
        # ------- compute_clusters START
        clusters = []
        for food in food_list:
            count = 0
            for rest_food in food_list:
                if self.get_maze_distance(food, rest_food) <= radius:
                    count += 1
            clusters.append((food, count))
        # -------- compute_clusters END
        best_food = None
        best_cluster_size = 0

        for food, size in clusters:
            if size > best_cluster_size:
                best_cluster_size = size
                best_food = food

        state = successor.get_agent_state(self.index)
        my_pos = state.get_position()
        if my_pos is None:
            return features
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        active_defenders = [a for a in defenders if a.scared_timer == 0]
        scared_defenders = [a for a in defenders if a.scared_timer > 0]
        is_chased = False
        closest_defender_dist = float('inf')
        if active_defenders:
            defender_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in active_defenders]
            closest_defender_dist = min(defender_dists)
            is_chased = closest_defender_dist <= 5

        if state.is_pacman and closest_defender_dist <= 5:
            features['ghost_proximity'] = 10 - closest_defender_dist
        
        features['score'] = self.get_score(successor)
        features['uneaten_food'] = len(food_list)

        if best_food is not None:
            distance = self.get_maze_distance(my_pos, best_food)
            features['distance_to_cluster'] = distance
            features['cluster_size'] = best_cluster_size

        if scared_defenders:
            min_scared_timer = min(a.scared_timer for a in scared_defenders)
            if min_scared_timer >= 5:
                features['return_home'] = 0
                features['dead_end'] = 0
                scared_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in scared_defenders]
                features['dist_to_scared_defender'] = min(scared_dists)
        else:
            carrying = state.num_carrying
            distance_to_home = self.get_maze_distance(my_pos, self.start)
            time_left = successor.data.timeleft
            time = 1200
            urgency = 1 - time_left / time
            if is_chased:
                capsules = self.get_capsules(successor)
                if capsules:
                    capsule_dists = [self.get_maze_distance(my_pos, capsule) for capsule in capsules]
                    features['dist_to_capsule'] = min(capsule_dists)
                else:
                    features['return_home'] = carrying * distance_to_home * urgency * 5
            else:
                features['return_home'] = carrying * distance_to_home * urgency

        if active_defenders:
            for defender in active_defenders:
                defender_pos = defender.get_position()
                if my_pos == defender_pos: features['walk_into_defender'] = 1
        if my_pos in self.dead_ends:
            depth = self.dead_ends[my_pos]
            if closest_defender_dist <= depth * 2:
                features['dead_end'] = 1

        if self.pos_history:
            count = self.pos_history.count(my_pos)
            features['reverse'] = count

        return features
    
    def get_weights(self, game_state, action):
        return {'score': 100,
                   'uneaten_food': -100,
                   'distance_to_cluster': -1,
                   'cluster_size': 1,
                   'return_home': -1,
                   'dead_end': -75,
                   'reverse': 0,
                   'ghost_proximity': -10,
                   'dist_to_capsule': -10,
                   'walk_into_defender': -100,
                   'dist_to_scared_defender': -15}