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

    #TODO: Consider set implementation (membership check in O(1) )
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
                self.debug_draw(not_wall, color=(224,33,216))

        while not queue.is_empty():
            not_wall = queue.pop()
            for x in neighbours[not_wall]:
                if x not in degree: continue
                degree[x] -= 1

                if degree[x] == 1 and x not in self.dead_ends:
                    self.dead_ends[x] = self.dead_ends[not_wall] + 1
                    queue.push(x)
                    self.debug_draw(x, color=(224,33,216))

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
    #TODO: exclude from computed clusters
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
        defense_midfield_x = middle_x - middle_x // 4 if self.red else middle_x + middle_x // 4
        result = []
        maze_height = game_state.data.layout.height

        blue_colrange = range(middle_x, defense_midfield_x)
        red_colrange = range(defense_midfield_x, middle_x)
        colrange = red_colrange if self.red else blue_colrange

        for col in colrange:
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
    def get_food_close_to_border(self, game_state):
        food_list = self.get_food(game_state).as_list()
        middle_x = (game_state.data.layout.width - 1) // 2 if self.red else game_state.data.layout.width // 2
        food_with_dist = []
        for food in food_list:
            dist = abs(food[0] - middle_x)
            food_with_dist.append((dist, food))
        food_with_dist.sort()
        return [food for _, food in food_with_dist]

    def get_border_dist(self, game_state, pos):
        middle_x = (game_state.data.layout.width - 1) // 2 if self.red else game_state.data.layout.width // 2
        height = game_state.data.layout.height
        border_cells = [(middle_x, y) for y in range(height-1) if not game_state.has_wall(middle_x,y)]
        if not border_cells: return 0
        return min(self.get_maze_distance(pos, bc) for bc in border_cells)

    def get_features(self, game_state, action):
        features = util.Counter()
        current_food = self.get_food_you_are_defending(game_state).as_list()

        # code bellow checks if our food is eaten and returns the closest position for which this is the case
        eaten = set(self.previous_food) - set(current_food)
        if eaten:
            #TODO: zie comment lijn 617
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
        #TODO: onderstaande variabelen meermaals gebruikt doorheen code
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        scared_timer = game_state.get_agent_state(self.index).scared_timer
        is_scared = scared_timer > 0

        retreat_threshold = scared_timer <= self.get_border_dist(game_state, my_pos)
        if is_scared:
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            active_defenders = [a for a in defenders if a.scared_timer == 0]
            is_chased = False
            #FIXME: unused variables
            closest_defender_dist = float('inf')
            carrying = successor.get_agent_state(self.index).num_carrying
            if active_defenders:
                defender_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in active_defenders]
                closest_defender_dist = min(defender_dists)
                is_chased = closest_defender_dist <= 5
            prev_pos = game_state.get_agent_state(self.index).get_position()
            if my_pos == self.start and prev_pos != self.start:
                features['dont_die'] = 1
            if scared_timer > retreat_threshold:
                border_food = self.get_food_close_to_border(game_state)
                if border_food:
                    dists = [self.get_maze_distance(my_pos, food) for food in border_food]
                    features['raid_food_dist'] = min(dists)
                if is_chased:
                    features['return_home'] = self.get_border_dist(game_state, my_pos)
            else:
                features['return_home'] = self.get_border_dist(game_state, my_pos)
        else:
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
            features['bottleneck_distance'] = min(bottleneck_dist) if bottleneck_dist else 0
            #FIXME: Code duplication with offensive reflex
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                features['capsules'] = len(capsules)
                capsule_dists = [self.get_maze_distance(my_pos, capsule) for capsule in capsules]
                features['dist_to_capsule'] = max(capsule_dists)


        #TODO: succesor_score feature and its weight from super are overwritten
        return features

    def get_weights(self, game_state, action):
        #TODO: zie in welke scope definieerbaar. Indien op andere plekken nuttig, hoger in de scope
        def is_scared():
            if game_state.get_agent_state(self.index).scared_timer > 0:
                return True
            else:
                return False
        if is_scared():
            return {'invader_distance': 5,
                    'trapped_invader_distance': 50,
                    'raid_food_dist': -10,
                    'return_home': -100,
                    'dont_die': -100}
        #FIXME: delete unused conditional weights
        invader_distance_w = -100 if not is_scared() else 5
        trapped_invader_distance_w = -150 if not is_scared() else 50
        return {'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -200,
                'trapped_invader_distance': -150,
                'stop': -100,
                'reverse': -2,
                'distance_to_last_eaten_food': -20,
                'bottleneck_distance': -15,
                'dist_to_capsule': -10,
                'capsules': 1000}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    #TODO: Als in eigen regio & invader binnen kleine radius ->> chap die man
    def register_initial_state(self, game_state):
        #FIXME: variabelen die in init kunnen mss in init
        super().register_initial_state(game_state)
        self.pos_history = []
        self.pos_hist_len = 4
        self.steps_on_own_half = 0
        self.initial_timeleft = 1200
        # bereken op voorhand de plekken waar we terug naar huis kunnen gaan
        self.home_positions = self._compute_home_positions(game_state)

    def _compute_home_positions(self, game_state):
        """
        Deze hulpprocedure berekent posities waar de agent terug naar home-base kan gaan.
        Wordt in de register_initial_state gerund en opgeslagen als een lijst.
        """
        layout = game_state.data.layout
        # TODO: Dit is een iets properdere versie dan wat ik voordien in de defensive agent heb gedaan.
        #       Misschien daar ook aanpassen zodat het wat cleaner leest/oogt
        # kolom vanaf waar we technisch gezien "home" zijn, afhankelijk van team.
        home_x = (layout.width - 1) // 2 if self.red else layout.width // 2
        #TODO: implementatie als een set in plaats van lijst voor snellere membership acces? Zou normaal toch geen duplicate coördinaten moeten bevatten
        return [
            (home_x, y)
            # de effectief bewandelbare maze rij-coördinaten zijn tussen 1 en de layout-hoogte -1 want muren aan de randen
            for y in range(1, layout.height - 1)
            if not game_state.has_wall(home_x, y)
        ]

    def _distance_to_home_position(self, pos):
        """
        interne hulpprocedure die gegeven home_positions de kortste afstand vindt naar home.
        Neemt positie tuple en geeft een afstand terug.
        """
        return min(self.get_maze_distance(pos, home_pos) for home_pos in self.home_positions)

    def _best_food_target(self, game_state, my_pos, food_list, defenders, radius=2):
        """
        Helper functie die helpt met bepalen wat de beste food-target is op dit moment.
        Gebeurt op basis van berekenen waar clusters van voedsel zich bevinden en hun grootte.
        Aangepaste Dijkstra procedure helpt met bepalen hoe 'veilig' het is om daar te geraken
        """
        # Zou normaal gezien niet moeten voorkomen gezien de game stopt bij de laatste 2, maar just in case want weet
        # nog niet zeker hoe de game logica hier in elkaar zit.
        if not food_list:
            return None, 0, 0

        clusters = []
        for food in food_list:
            count = 0
            for rest_food in food_list:
                if self.get_maze_distance(food, rest_food) <= radius:
                    count += 1
            clusters.append((food, count))

        best_food = None
        best_cluster_size = 0
        best_cost = float('inf')
        # Ga voor elke food na welke de beste is op basis van de dijkstra measure
        for food, size in clusters:
            path_cost = self.dijkstra_distance(game_state, my_pos, food, defenders)
            score = path_cost - (size * 2)
            if score < best_cost:
                best_cost = score
                best_food = food
                best_cluster_size = size

        return best_food, best_cluster_size, best_cost
    #TODO make private
    def dijkstra_distance(self, game_state, start, target, defenders, danger_radius=5, penalty_weight=10):
        """
        Adaptatie van dijkstra's algoritme, geef kortste afstand tot een target tenzij er (gevaarlijke) adversaries zijn: penaliseer in dat geval paden
        die gevaarlijk zijn door een aangepaste afstand.
        """
        # Skip Dijkstra if no active defenders/ defenders too far away
        if not defenders: 
            return self.get_maze_distance(start, target)
        
        min_defender_dist = min(self.get_maze_distance(start, defender.get_position()) for defender in defenders)
        if min_defender_dist > danger_radius:
            #TODO: zie opmerking als elders: zijn we in danger vanaf een manhatten distance kleiner of gelijk aan 5?
            return self.get_maze_distance(start, target)
        
        walls = game_state.get_walls()

        def penalty(cell):
            min_dist = min(self.get_maze_distance(cell, defender.get_position()) for defender in defenders)
            return max(0, danger_radius - min_dist) * penalty_weight # simple function, the closer a defender, the worse (can be finetuned)

        pq = PriorityQueue()
        pq.push(start, 0)
        visited = set()
        costs = {start: 0}

        #Start Dijkstra
        while not pq.is_empty():
            cell = pq.pop()
            if cell in visited: continue
            visited.add(cell)
            if cell == target:
                return costs[cell]
            x, y = int(cell[0]), int(cell[1])
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                newx, newy = x + dx, y + dy
                neighbour = (newx, newy)
                if not walls[newx][newy] and neighbour not in visited:
                    new_cost = costs[cell] + 1 + penalty(neighbour)
                    if neighbour not in costs or new_cost < costs[neighbour]:
                        costs[neighbour] = new_cost
                        pq.update(neighbour, new_cost)
        return float('inf') #safety fallback, when the entire pq has been exhausted w/o path to target
    
    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        state = game_state.get_agent_state(self.index)

        if my_pos is not None:
            # record positie
            self.pos_history.append(my_pos)
            # we houden slechts een bepaald aantal posities vast
            if len(self.pos_history) > self.pos_hist_len:
                self.pos_history.pop(0)

        # counter gebruikt om bij te houden hoe lang pacman aan zijn eigen kant gebruikt
        # We willen een tradeoff hebben dat pacman soms aan zijn kant blijft om de ocassionele vijand te capturen
        # maar dit mag ook niet te lang oplopen zodat het objectief dots eten blijft
        if not state.is_pacman:
            self.steps_on_own_half += 1
        else:
            self.steps_on_own_half = 0

        actions = game_state.get_legal_actions(self.index)
        # filteren van stop, is zeer zelden een nuttige actie voor de offensive
        legal_actions = [action for action in actions if action != Directions.STOP]

        # Anti-oscillation
        if len(self.pos_history) >= 4:
            # detectie: tussen twee posities geoscilleerd
            if (self.pos_history[-1] == self.pos_history[-3] and
                    self.pos_history[-2] == self.pos_history[-4]):
                current_direction = game_state.get_agent_state(self.index).configuration.direction
                # check voor legale acties die
                non_reverse = [
                    action for action in legal_actions
                    if action != Directions.REVERSE[current_direction]
                ]
                if non_reverse:
                    return random.choice(non_reverse)
               # FIXME: Else return random action whatsover? Beter om dood te gaan en opnieuw te beginnen dan te blijven in een sink-state?  

        # TODO: code duplicatie van parent, maar bovenstaande moet eerder gerund worden
        values = [self.evaluate(game_state, action) for action in legal_actions]
        max_value = max(values)
        best_actions = [action for action, value in zip(legal_actions, values) if value == max_value]

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if best_action is not None:
                return best_action

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        state = successor.get_agent_state(self.index)
        my_pos = state.get_position()
        if my_pos is None:
            return features

        prev_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        # defenders zijn tegenstanders aan de overkant die hun food verdedigen
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        active_defenders = [a for a in defenders if a.scared_timer == 0]
        scared_defenders = [a for a in defenders if a.scared_timer > 0]
        # bijhouden van invaders: als we onderweg zijn naar de overkant, willen we soms een vijand onderweg capturen
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # we zijn chased als er actieve defenders zijn in onze directe observeerbare radius (zij zien ons)
        is_chased = False
        closest_defender_dist = float('inf')
        if active_defenders:
            #TODO: we gebruiken hier maze distance, maar observeerbaarheid hangt af van manhattan distance
            #      het kan zijn dat de maze distance hier tekkortschiet, enemies kunnen ons al eerder zien en beginnen chasen?
            defender_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in active_defenders]
            closest_defender_dist = min(defender_dists)
            is_chased = closest_defender_dist <= 5

        #TODO: document reasoning voor 5 en 10
        if state.is_pacman and closest_defender_dist <= 5:
            features['ghost_proximity'] = 10 - closest_defender_dist

        if my_pos == self.start and prev_pos != self.start:
            features['dont_die'] = 1
        # directe reward, heeft invloed op het incashen van eten
        features['score'] = self.get_score(successor)
        # geeft een meer globale druk om te eten in plaats van treuzelen of andere niet-eet acties
        features['uneaten_food'] = len(food_list)

        best_food, best_cluster_size, best_cost = self._best_food_target(
            successor,
            my_pos,
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

        carrying = state.num_carrying
        time_left = successor.data.timeleft
        # urgency vb: low urgency begin van spel → 1 - (1200/1200) = 0
        # hoge urgency einde van spel → 1 - (400/1200) = 0.7 (1 is hoogste urgency)
        urgency = 1 - (time_left / self.initial_timeleft)
        # in mentaliteit van evaluatie ook te zien als maat voor "hoe snel kan ik food in cashen?"
        distance_to_home = self._distance_to_home_position(my_pos)

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
        prev_distance_to_home = self._distance_to_home_position(prev_pos)
        if carrying > 0 and time_left <= prev_distance_to_home + 50:
            if distance_to_home < prev_distance_to_home:
                features['cash_in_now'] = 1

        # capsules worden interessanter/belangrijker wanneer in chase of wanneer
        # er meer voedsel in bezit is (hogere risico situatie)
        # FIXME: mogelijke bug van capsule niet eten: als die eet dan wordt de distance naar een eventuele andere capsule ineens zeer groot, en geeft
        #       een penalty voor de volgende actie
        capsules = self.get_capsules(successor)
        if capsules:
            capsule_dists = [self.dijkstra_distance(successor, my_pos, capsule, active_defenders) for capsule in capsules]
            features['dist_to_capsule'] = min(capsule_dists)
            if is_chased or carrying >= 4:
                # hoe meer je draagt, hoe meer je te verliezen hebt, hoe interessanter het wordt om defense van de tegenstander uit te schakelen
                features['capsule_pressure'] = 1 * carrying

        if scared_defenders:
            min_scared_timer = min(a.scared_timer for a in scared_defenders)
            if min_scared_timer >= 4:
                features['dead_end'] = 0
                prev_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                prev_scared = [a for a in prev_enemies if not a.is_pacman and a.scared_timer > 0 and a.get_position() is not None]
                prev_scared_positions = [a.get_position() for a in prev_scared]
                if my_pos in prev_scared_positions:
                    features['ate_scared_ghost'] = 1
                else:
                    scared_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in scared_defenders]
                    features['dist_to_scared_defender'] = min(scared_dists)

        if active_defenders:
            for defender in active_defenders:
                defender_pos = defender.get_position()
                if my_pos == defender_pos:
                    features['walk_into_defender'] = 1

        if my_pos in self.dead_ends:
            depth = self.dead_ends[my_pos]
            if closest_defender_dist <= depth * 2:
                features['dead_end'] = 1

        # FIXME: reverse is deprecated: Zie anti oscillatie in choose action
        #       weight bij submission van agent ook 0 dus niet gebruikt
        if self.pos_history:
            count = self.pos_history.count(my_pos)
            features['reverse'] = count

        if not state.is_pacman:
            # manier om 'camping' op eigen grondgebied tegen te gaan
            features['steps_on_own_half'] = self.steps_on_own_half
            if invaders:
                invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                min_invader_dist = min(invader_distances)
                if min_invader_dist <= 3:
                    # voor goedkope defensieve acties als er vijand op eigen terrein is
                    features['close_invader_distance'] = min_invader_dist

        return features

    def get_weights(self, game_state, action):
        # for mental sanity als door de bomen het bos niet zien
        # Mogelijke interacties tussen features:
        #   motivatie voor eten versus 'veilig spelen':
        #       uneaten_food
        #       cluster_size
        #       distance_to_cluster
        #       → versterk deze voor eet-motivatie
        #       ghost_proximity
        #       dead_end
        #       walk_into_defender
        #       dont_die
        #       → versterk deze voor veiligheid
        #    motivatie voor terug naar huis gaan:
        #       return_home als meer globale gradueel effect en cash_in_now als harde trigger in end-game
        #    defense op eigen kant:
        #       close_invader_distance
        #       steps_on_own_half
        #       → om effect toe te nemen: verminder close_invader,
        #       voor duur pas penalisatie van steps on own half aan


        return {'score': 100,
                   'uneaten_food': -150,
                   'distance_to_cluster': -10,
                   'cluster_size': 5,
                   'return_home': -4,
                   'cash_in_now': 150,
                   'dead_end': -75,
                   'reverse': 0,
                   'ghost_proximity': -10,
                   'dist_to_capsule': -18,
                   'capsule_pressure': 40,
                   'walk_into_defender': -100,
                   'dist_to_scared_defender': -2,
                   'ate_scared_ghost': 5,
                   'dont_die': -1000,
                   'steps_on_own_half': -3,
                   'close_invader_distance': -12}
