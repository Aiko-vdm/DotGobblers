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
from util import PriorityQueue
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

class ReflexCaptureAgent(CaptureAgent):
    """A base class for reflex agents that choose score-maximizing actions."""
    def __init__(self, index, time_for_computing=.1):
        """
        start -- start position of the agent at the beginning of the game
        cardinal_distances -- the four cardinal direction offsets used for neighbour lookups
        walls -- grid of wall positions for the current maze layout
        dead_ends -- maps each dead-end cell to its depth (populated by _compute_dead_ends)
        middle_x -- x-coordinate of the border between the two teams' halves
        observable_distance -- from this distance, you can either chase or be chased due to observable distance
        """
        super().__init__(index, time_for_computing)
        self.start = None
        self.cardinal_distances = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.walls = None
        self.dead_ends = {}
        self.middle_x = None
        self.observable_distance = 5


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        CaptureAgent.register_initial_state(self, game_state)
        self._compute_dead_ends()
        self.middle_x = (game_state.data.layout.width - 1) // 2 if self.red else game_state.data.layout.width // 2

    def _get_walkable_neighbours(self, cell):
        """ Return all non-wall positions directly neighboring cell."""
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in self.cardinal_distances if not self.walls[x + dx][y + dy]]
    
    def _compute_dead_ends(self):
        """Identify dead-end cells in a maze and records how deep each one is.

        A cell with depth 1 only has one neighboring walkable cell. A cell with depth n is a corridor of length n.
        Uses a BFS-like algorithm: fill the queue with all cells with depth 1 (dead-end tips), 
        reduce neighboring degrees and propagate until no further dead ends are found.
        """
        neighbours = {}
        degree = {}

        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if self.walls[x][y]:
                    continue
                walkable_cell = (x, y)
                list_of_neighbours = self._get_walkable_neighbours(walkable_cell)
                neighbours[walkable_cell] = list_of_neighbours
                degree[walkable_cell] = len(list_of_neighbours)

        queue = Queue()
        for walkable_cell in degree:
            if degree[walkable_cell] == 1:
                queue.push(walkable_cell)
                self.dead_ends[walkable_cell] = 1
                self.debug_draw(walkable_cell, color=(224,33,216))

        while not queue.is_empty():
            walkable_cell = queue.pop()
            for neighbour in neighbours[walkable_cell]:
                if neighbour not in degree:
                    continue
                degree[neighbour] -= 1
                # "remove" dead-end tip from the maze, if neighbour's degree drops to 1 as a result, it becomes a new dead-end tip 
                if degree[neighbour] == 1 and neighbour not in self.dead_ends: 
                    self.dead_ends[neighbour] = self.dead_ends[walkable_cell] + 1
                    queue.push(neighbour)
                    self.debug_draw(neighbour, color=(224,33,216))
    
    def _get_enemy_states(self, game_state):
        """ Return agent states of all opponents.  """
        return [game_state.get_agent_state(opponent) for opponent in self.get_opponents(game_state)]

    def _get_active_defenders(self, enemy_states):
        """ Return enemy ghosts that are visible and not scared.  """
        return [enemy for enemy in enemy_states 
                if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer == 0]

    def _get_scared_defenders(self, enemy_states):
        """ Return enemy ghosts that are visible and currently scared.  """
        return [enemy for enemy in enemy_states
                if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer > 0]
    
    def _get_invaders(self, enemy_states):
        """ Return enemy ghosts that are visible and currently scared.  """
        return [enemy for enemy in enemy_states if enemy.is_pacman and enemy.get_position() is not None]
    
    def _closest_defender_distance(self,my_position,active_defenders):
        """ Return the maze distance to the nearest active defender, or inf when none are visible/exist. """
        if not active_defenders:
            return float('inf')
        return min(self.get_maze_distance(my_position, defender.get_position()) for defender in active_defenders)
    
    def _is_chased(self, my_position, active_defenders):
       """ Return True when at least one active defender is within observable range."""
       return self._closest_defender_distance(my_position, active_defenders) <= self.observable_distance
     
    def choose_action(self, game_state):
        """ Picks among the actions with the highest Q(s,a). """
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
        # Endgame: when only 2 or fewer food dots remain, picks action that take agent as close as possible to start position
        if food_left <= 2:
            return min(actions, key=lambda action: self.get_maze_distance(self.start, self.get_successor(game_state, action).get_agent_position(self.index)))
    
        return random.choice(self._evaluate_actions(game_state,actions))
    
    def _evaluate_actions(self, game_state, actions):
        """ Return all actions form actions with the highest Q-value """
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        return [action for action, value in zip(actions, values) if value == max_value]
    
    def get_successor(self, game_state, action):
        """Finds the next successor which is a grid position (location tuple)."""
        successor = game_state.generate_successor(self.index, action)
        position = successor.get_agent_state(self.index).get_position()
        if position != nearest_point(position):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """Compute a linear combination of features and feature weights."""
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """Return a counter of features for the state"""
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """Return a counter of weights for the features of the state"""

        return {'successor_score': 1.0}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. When an invader eats a capsule and the defensive reflex agent turns scared,
    it crosses the border into the opponents part of the maze and tries to eat some food dots for as long it's scared. 
    After its scared_timer runs out, the defensive agent returns to its normal defensive behaviour.
    """
  
    def __init__(self, index, time_for_computing=.1):
        """
        bottleneck_positions -- narrow horizontal passages on team side used as priority patrol positions
        last_eaten_food -- position of the most recently eaten food dot on team side, used to track the invader
        previous_food -- snapshot of the defended food list from the previous turn, used to detect eaten food dots
        """
        super().__init__(index, time_for_computing)
        self.bottleneck_positions = None
        self.last_eaten_food = None
        self.previous_food = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
        self._find_bottlenecks(game_state)
        # Flush any memory from previous games, see commit 28a1fa for more details
        self.last_eaten_food = None
        self._draw_bottlenecks()

    def _draw_bottlenecks(self):
        """Draw the bottleneck locations"""
        for bottleneck in self.bottleneck_positions:
            self.debug_draw(bottleneck, color=(158,224,32))
    
    def _position_is_gate(self,row,column,game_state):
        """Returns True if the cell is a narrow horizontal passage: walkable horizontally but walled off vertically."""
        return all([not game_state.has_wall(column, row),
                    not game_state.has_wall(column + 1, row),
                    not game_state.has_wall(column - 1, row),
                    game_state.has_wall(column, row - 1),
                    game_state.has_wall(column, row + 1)])
    
    def _find_bottlenecks(self, game_state):
        """Look for bottleneck positions on team side and assign to self.bottleneck_positions

        Bottleneck positions are narrow horizontal passages that may be interesting for defense.
        These are detected by the position_is_gate function in the space between the middle and quarter-line
        of the maze.
        """
        quarterfield_x = self.middle_x - (self.middle_x // 4) if self.red else self.middle_x + (self.middle_x // 4)
        result = []
        maze_height = game_state.data.layout.height
        blue_columnrange = range(self.middle_x, quarterfield_x)
        red_columnrange = range(quarterfield_x, self.middle_x)
        columnrange = red_columnrange if self.red else blue_columnrange
        
        for column in columnrange:
            for row in range(1,maze_height - 1 ):
                if self._position_is_gate(row,column,game_state):
                    result.append((column, row))
        self.bottleneck_positions = result

    def _get_food_close_to_border(self, game_state):
        """Returns all food pellets on the opponent's side sorted by distance to the border, closest first."""
        food_list = self.get_food(game_state).as_list()
        food_with_dist = []
        for food in food_list:
            dist = abs(food[0] - self.middle_x)
            food_with_dist.append((dist, food))
        food_with_dist.sort()
        return [food for _, food in food_with_dist]
   
    def _get_border_dist(self, game_state, position):
        """Returns the maze distance from position to the nearest walkable cell on the border between the two teams' halves."""
        height = game_state.data.layout.height
        border_cells = [(self.middle_x, y) for y in range(height-1) if not game_state.has_wall(self.middle_x,y)]
        if not border_cells:
            return 0
        return min(self.get_maze_distance(position, border_cell) for border_cell in border_cells)

    def get_features(self, game_state, action):
        #TODO: opslitsen in hulpprocedures? Of op zijn minst code blocks benoemen
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
            self.last_eaten_food = closest
        self.previous_food = current_food

        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_position = my_state.get_position()
        scared_timer = game_state.get_agent_state(self.index).scared_timer
        is_scared = scared_timer > 0
        should_retreat = scared_timer <= self._get_border_dist(game_state, my_position)
        if is_scared:
            enemy_states = self._get_enemy_states(successor)
            active_defenders = self._get_active_defenders(enemy_states)
            is_chased = self._is_chased(my_position, active_defenders)

            previous_position = game_state.get_agent_state(self.index).get_position()
            if my_position == self.start and previous_position != self.start:
                features['dont_die'] = 1
            if scared_timer > should_retreat: #FIXME: bug
                border_food = self._get_food_close_to_border(game_state)
                if border_food:
                    distances = [self.get_maze_distance(my_position, food) for food in border_food]
                    features['raid_food_dist'] = min(distances)
                if is_chased:
                    features['return_home'] = self._get_border_dist(game_state, my_position)
            else:
                features['return_home'] = self._get_border_dist(game_state, my_position)
        else:
            # Computes whether we're on defense (1) or offense (0)
            features['on_defense'] = 1
            if my_state.is_pacman:
                features['on_defense'] = 0

            # Computes distance to invaders we can see
            enemy_states = self._get_enemy_states(successor)
            invaders = self._get_invaders(enemy_states)
            features['num_invaders'] = len(invaders)
            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_position, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
                # Of those we see, how many are trapped in dead ends
                # FIXME: bug
                dist_trapped = [self.get_maze_distance(my_position, a.get_position()) for a in enemy_states if a in self.dead_ends]
                features['trapped_invader_distance'] = min(dist_trapped) if len(dist_trapped) > 0 else 0

            if action == Directions.STOP:
                features['stop'] = 1

            reverse_action = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == reverse_action:
                features['reverse'] = 1

            if len(invaders) == 0 and self.last_eaten_food is not None:
                distance = self.get_maze_distance(my_position, self.last_eaten_food)
                features['distance_to_last_eaten_food'] = distance

            #distance to a bottleneck
            bottleneck_distance = [self.get_maze_distance(my_position, bottleneck) for bottleneck in self.bottleneck_positions]
            features['bottleneck_distance'] = min(bottleneck_distance) if bottleneck_distance else 0
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                features['capsules'] = len(capsules)
                capsule_distances = [self.get_maze_distance(my_position, capsule) for capsule in capsules]
                features['distance_to_capsule'] = max(capsule_distances)


        return features

    def get_weights(self, game_state, action):

        # weights for when the agent is scared
        if game_state.get_agent_state(self.index).scared_timer > 0:
            return {'invader_distance': 5,
                    'trapped_invader_distance': 50,
                    'raid_food_dist': -10,
                    'return_home': -100,
                    'dont_die': -100}
        
        return {'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -200,
                'trapped_invader_distance': -150,
                'stop': -100,
                'reverse': -2,
                'distance_to_last_eaten_food': -20,
                'bottleneck_distance': -15,
                'distance_to_capsule': -10,
                'capsules': 1000}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food and avoids defenders.
  """
    def __init__(self, index, time_for_computing=.1):
        """
        position_history -- a list that stores the last n positions with n defined by position_history_length, used for anti-oscillation
        position_history_length -- the length of the last n positions stored in position_history
        home_positions -- legal positions where the agent can return home
        steps_on_own_half -- number of steps taken on team side
        initial_time_left -- the total game time at the start of a game
        food_cluster_radius -- the radius used to detect a cluster
        cluster_size_score_factor -- score used to reward bigger clusters
        endgame_home_slack -- time offset given for endgame strategy. Increase tot start the endgame strategy earlier.
        invader_close_distance -- max distance to attack enemy invaders when on own side
        dead_end_danger_factor -- a number multiplied with the depth of dead ends to give escape margin from dead-ends

        """
        super().__init__(index, time_for_computing)
        self.position_history = []
        self.position_history_length = 4
        self.home_positions = None
        self.steps_on_own_half = 0
        self.initial_time_left = 1200
        self.food_cluster_radius = 2
        self.cluster_size_score_factor = 2
        self.endgame_home_slack = 50
        self.invader_close_distance = 3
        self.dead_end_danger_factor = 1.5


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.home_positions = self._compute_home_positions(game_state)
        # Flush memory from previous games, see commit 28a1fa for more details
        self.position_history = []
        self.steps_on_own_half = 0


    def _compute_home_positions(self, game_state):
        """
        Computes legal positions where the agent can return home.
        Runned in the register_initial_state and stored as a list.
        """
        layout = game_state.data.layout
        return [(self.middle_x, y) for y in range(1, layout.height - 1) if not game_state.has_wall(self.middle_x, y)]

    def _distance_to_home_position(self, position):
        """
        Finds the shortest distance from position to home given home_positions.
        """
        return min(self.get_maze_distance(position, home_pos) for home_pos in self.home_positions)

    def _best_food_target(self, game_state, my_position, food_list, defenders):
        """
        Determines the best food target at this moment:
        calculates the location and size of food clusters + 
        Dijkstra procedure helps determine how 'safe' it is to reach that location.
        """
        if not food_list:
            return None, 0, 0

        clusters = []
        for food in food_list:
            count = 0
            for rest_food in food_list:
                if self.get_maze_distance(food, rest_food) <= self.food_cluster_radius:
                    count += 1
            clusters.append((food, count))

        best_food = None
        best_cluster_size = 0
        best_cost = float('inf')
        # For each food, determine which is the best based on the Dijkstra measure.
        for food, size in clusters:
            path_cost = self._dijkstra_distance(my_position, food, defenders)
            score = path_cost - (size * self.cluster_size_score_factor)
            if score < best_cost:
                best_cost = score
                best_food = food
                best_cluster_size = size

        return best_food, best_cluster_size, best_cost
    
    def _cell_danger_penalty(self, cell, defenders, danger_radius, penalty_weight):
        """Compute a danger penalty for a given cell based on the proximity of active enemy defenders.

        The closer a defender is to the cell, the bigger the penalty. 
        Return 0 if there's no active defender within the danger_radius of the cell.
        """
        min_dist = min(self.get_maze_distance(cell, defender.get_position()) for defender in defenders)
        return max(0, danger_radius - min_dist) * penalty_weight

    def _dijkstra_distance(self, start, target, defenders, danger_radius=5, penalty_weight=10):
        """Compute the cost of the shortest path from start to target using a penalty-weighted Dijkstra.

         Paths passing near active defenders are penalized proportionally to their proximity.
         Fall back to maze distance when there are no defenders or all of the defenders are outside the danger_radius of start.
        """
        unreachable = float('inf')

        no_threat_nearby = (not defenders or 
                            min(self.get_maze_distance(start, defender.get_position()) for defender in defenders) > danger_radius)
        if no_threat_nearby:
            return self.get_maze_distance(start, target)

        pq = PriorityQueue()
        pq.push(start, 0)
        visited = set()
        costs = {start: 0}

        while not pq.is_empty():
            current_cell = pq.pop()
            if current_cell in visited:
                continue
            visited.add(current_cell)
            if current_cell == target:
                return costs[current_cell]
            x, y = int(current_cell[0]), int(current_cell[1])
            for dx, dy in self.cardinal_distances:
                newx, newy = x + dx, y + dy
                neighbour = (newx, newy)
                if not self.walls[newx][newy] and neighbour not in visited:
                    new_cost = costs[current_cell] + self._cell_danger_penalty(neighbour, defenders, danger_radius, penalty_weight) + 1
                    if neighbour not in costs or new_cost < costs[neighbour]:
                        costs[neighbour] = new_cost
                        pq.update(neighbour, new_cost)
        return unreachable 
    
    def choose_action(self, game_state):
        """ 
        Picks among the actions with the highest Q(s,a). 
        Extends the parent's choose_action with three additional behaviours:
            - tracks steps spent on own half to discourage camping
            - filters out STOP from the legal actions
            - tracks position history to detect and break oscillation between two positions
        """
        my_position = game_state.get_agent_state(self.index).get_position()
        state = game_state.get_agent_state(self.index)
        # update position history
        if my_position is not None:
            self.position_history.append(my_position)
            if len(self.position_history) > self.position_history_length:
                self.position_history.pop(0)

        # count steps on own side
        if not state.is_pacman:
            self.steps_on_own_half += 1
        else:
            self.steps_on_own_half = 0

        actions = game_state.get_legal_actions(self.index)
        legal_actions = [action for action in actions if action != Directions.STOP]

        # Anti-oscillation
        if len(self.position_history) >= self.position_history_length:
            # detect when we oscillate between two positions and if so
            if (self.position_history[-1] == self.position_history[-3] and
                    self.position_history[-2] == self.position_history[-4]):
                current_direction = game_state.get_agent_state(self.index).configuration.direction
                # check voor legale acties die
                non_reverse = [
                    action for action in legal_actions
                    if action != Directions.REVERSE[current_direction]
                ]
                if non_reverse:
                    return random.choice(non_reverse)

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            return min(legal_actions, key=lambda action: self.get_maze_distance(self.start, self.get_successor(game_state, action).get_agent_position(self.index)))

        return random.choice(self._evaluate_actions(game_state, legal_actions))

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_state = successor.get_agent_state(self.index)
        my_position = my_state.get_position()
        if my_position is None:
            return features
      
        previous_position = game_state.get_agent_state(self.index).get_position()
        enemy_states = self._get_enemy_states(successor)
        active_defenders = self._get_active_defenders(enemy_states)
        scared_defenders = self._get_scared_defenders(enemy_states)
        invaders = self._get_invaders(enemy_states)

        closest_defender_distance = self._closest_defender_distance(my_position, active_defenders)
        is_chased = self._is_chased(my_position, active_defenders)
        
        if my_state.is_pacman and closest_defender_distance <= self.observable_distance:
            features['ghost_proximity'] = self.observable_distance * 2 - closest_defender_distance

        if my_position == self.start and previous_position != self.start:
            features['dont_die'] = 1
        # directe reward, heeft invloed op het incashen van eten
        features['score'] = self.get_score(successor)
        # geeft een meer globale druk om te eten in plaats van treuzelen of andere niet-eet acties
        features['uneaten_food'] = len(food_list)

        best_food, best_cluster_size, best_cost = self._best_food_target(
            successor,
            my_position,
            food_list,
            active_defenders,
        )
        if best_food is not None:
            # opgelet: afstand tot eten is meer 'defender' aware wegens specifieke dijkstra_distance implementatie
            # redeneren over onderstaande best door beiden features in tanden te nemen:
                # Actie a gaat naar kleine cluster --> distance = 6, size = 2
                # Actie b gaat naar grote cluster maar iets verder weg --> distance = 8, size = 5
                # eval a = -1 * 6 + 1 * 2 = -4
                # eval b = -1 * 8 + 1 * 5 = -3
                # verhogen van cluster size zal de 'tradeoff' met de afstand beïnvloeden
            features['distance_to_cluster'] = best_cost
            features['cluster_size'] = best_cluster_size

        carrying = my_state.num_carrying
        time_left = successor.data.timeleft
        # urgency vb: low urgency begin van spel → 1 - (1200/1200) = 0
        # hoge urgency einde van spel → 1 - (400/1200) = 0.7 (1 is hoogste urgency)
        urgency = 1 - (time_left / self.initial_time_left)
        # in mentaliteit van evaluatie ook te zien als maat voor "hoe snel kan ik food in cashen?"
        distance_to_home = self._distance_to_home_position(my_position)

        # meer food in bezit + langere afstand verhogen druk voor return home
        # urgency kan multipliceren tot x3
        features['return_home'] = carrying * distance_to_home * (1 + (2 * urgency))
        
        # manier om een finale sprint in de end game te motiveren.
        # idee: er zijn 2 situationele condities:
        #   Draag ik eten bij me?
        #   Is mijn tijd aan het verlopen relatief tot over mijn afstand naar huis?
        #       (we willen niet zomaar op basis van tijd triggeren, want je heb wel of geen tijd afhankelijk van waar je je bevindt)
        #       (we doen de afstand plust een 'slack window', want gewoon de afstand kan te nipt zijn gezien er onderweg nog obstakels kunnen voordoen)
        # zo ja: kijken we naar de actie: brengt het me dichter bij huis? Zo ja, cash in now!!
        prev_distance_to_home = self._distance_to_home_position(previous_position)
        if carrying > 0 and time_left <= prev_distance_to_home + self.endgame_home_slack:
            if distance_to_home < prev_distance_to_home:
                features['cash_in_now'] = 1

        # capsules worden interessanter/belangrijker wanneer in chase of wanneer
        # er meer voedsel in bezit is (hogere risico situatie)
        capsules = self.get_capsules(successor)
        if capsules:
            capsule_distance = [self._dijkstra_distance(my_position, capsule, active_defenders) for capsule in capsules]
            features['distance_to_capsule'] = min(capsule_distance)
            #TODO: magic constant 4
            if is_chased or carrying >= 4:
                # hoe meer je draagt, hoe meer je te verliezen hebt, hoe interessanter het wordt om defense van de tegenstander uit te schakelen
                features['capsule_pressure'] = carrying

        if scared_defenders:
            lowest_scared_timer = min(a.scared_timer for a in scared_defenders)
            #TODO magic constant 4
            if lowest_scared_timer >= 4:
                # geen penalty voor in een dead end zolang defenders scared zijn
                features['dead_end'] = 0
                previous_enemy_states = self._get_enemy_states(game_state)
                previous_scared_positions = [enemy.get_position() for enemy in self._get_scared_defenders(previous_enemy_states)]
                if my_position in previous_scared_positions:
                    features['ate_scared_ghost'] = 1
                else:
                    scared_dists = [self.get_maze_distance(my_position, a.get_position()) for a in scared_defenders]
                    features['distance_to_scared_defender'] = min(scared_dists)

        if active_defenders:
            for defender in active_defenders:
                defender_position = defender.get_position()
                if my_position == defender_position:
                    features['walk_into_defender'] = 1

        if my_position in self.dead_ends:
            depth = self.dead_ends[my_position]
            if closest_defender_distance <= depth * self.dead_end_danger_factor:
                features['dead_end'] = 1


        if not my_state.is_pacman:
            # manier om 'camping' op eigen grondgebied tegen te gaan
            features['steps_on_own_half'] = self.steps_on_own_half
            if invaders:
                invader_distances = [self.get_maze_distance(my_position, a.get_position()) for a in invaders]
                min_invader_distance = min(invader_distances)
                if min_invader_distance <= self.invader_close_distance:
                    # voor goedkope defensieve acties als er vijand op eigen terrein is
                    features['close_invader_distance'] = min_invader_distance

        return features

    def get_weights(self, game_state, action):
        return {'score': 100,
                   'uneaten_food': -150,
                   'distance_to_cluster': -10,
                   'cluster_size': 5,
                   'return_home': -4,
                   'cash_in_now': 150,
                   'dead_end': -75,
                   'ghost_proximity': -10,
                   'distance_to_capsule': -18,
                   'capsule_pressure': 40,
                   'walk_into_defender': -100,
                   'distance_to_scared_defender': -2,
                   'ate_scared_ghost': 5,
                   'dont_die': -1000,
                   'steps_on_own_half': -3,
                   'close_invader_distance': -12}
