# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random,util,math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from PIL import Image
from tensorflow.keras.utils import img_to_array

class QLearningAgent(ReinforcementAgent):
  
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter()
        self.ApproxState = util.Counter()
        self.trace = util.Counter()
        self.menace = 0
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        ################ 
        ApproxState = self.transformState(state)
        legalActions = self.getLegalActions(state)
        max_value = None

        """
        # SARSA :
        if util.flipCoin(self.epsilon):
              max_value = self.getQValue(ApproxState, random.choice(legalActions))
        else:
            max_value = self.getQValue(ApproxState, legalActions[0])
            for action in legalActions[1:] :
                  Q = self.getQValue(ApproxState, action)
                  if Q > max_value:
                        max_value = Q
        return max_value
        """         

        #Q-Learning:
        max_value = self.getQValue(ApproxState, legalActions[0])
        for action in legalActions[1:] :
              Q = self.getQValue(ApproxState, action)
              if Q > max_value:
                    max_value = Q
        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        ApproxState = self.transformState(state)
        
        if len(legalActions) == 0:
            return None
        start = random.randint(0, max(0,len(legalActions)-1))
        max_action = legalActions[start]
        max_action_value = self.getQValue(ApproxState, legalActions[start])
        for action in legalActions[0:]:
              Q = self.getQValue(ApproxState, action)
              if Q > max_action_value:
                    max_action_value = Q
                    max_action = action
        return max_action 

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
              action = random.choice(legalActions)
        else:
              action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        
        ApproxState = self.transformState(state)
        
        """if self.menace == 1:
            print(state)
            print("fantome: ", ApproxState[5:9], "   piege: ", ApproxState[9], "   conseil: ", ApproxState[4])
            print(reward)
            print("alpha =    ", self.alpha, "   discount = ", self.discount)
            print('')"""

        self.Qvalues[(ApproxState, action)] = self.getQValue(ApproxState, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(ApproxState,action) )

        """q_update = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(ApproxState, action)
        self.trace[(ApproxState, action)] = 1
        for s, a in self.trace.keys():
            self.Qvalues[(s, a)] = self.getQValue(s, a) + self.alpha * q_update * self.trace[(s, a)]
            self.trace[(s, a)] = self.discount * self.alpha * self.trace[(s, a)] # trace_gamma"""

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def transformState(self, state):
        #return state

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        newstate = []

        voisins = [(x,y+1),(x-1,y),(x,y-1),(x+1,y)]
        
        LegalNeighbors = Actions.getLegalNeighbors((x, y), walls)

        liste_voisins_valides = []
        # s1 à s4
        for pos in voisins:
            if pos in LegalNeighbors:
                newstate.append(0)
                liste_voisins_valides.append(pos)
            else:
                newstate.append(1)
                liste_voisins_valides.append(None)

        self.voisins = liste_voisins_valides

        # s5 à s10
        # x est la distance jusqu'à laquelle on indique à notre agent qu'il y a un fantôme menaçant
        # y // // // fantôme mangeable
        distance_fantomes = (7,4)

        # je les groupe puisqu'on utilise la même fonction de recherche, gourmande en calcul
        newstate = newstate + self.getS5S10(food, walls, ghosts, (x,y), distance_fantomes, liste_voisins_valides, state)
        result = tuple(newstate)

        #result = tuple(self.getS5S10(food, walls, ghosts, (x,y), distance_fantomes, liste_voisins_valides, state))
        return result

    def closestGhost(self, pos_pacman, pos_check, ghosts, walls, state, dist_max):
      """
      closestFood -- this is similar to the function that we have
      worked on in the search project; here its all in one place
      """

      fringe = [(pos_check[0], pos_check[1], 0)]
      expanded = set()
      expanded.add(pos_pacman)
      while fringe:
          pos_x, pos_y, dist = fringe.pop(0)
          if (pos_x, pos_y) in expanded or dist > dist_max:
              continue

          expanded.add((pos_x, pos_y))
          # if we find a ghosts at this location then exit
          if (pos_x,pos_y) in ghosts:
              if state.data.agentStates[1].scaredTimer > 0:
                  return dist, 1
              return dist, 0
              
          # otherwise spread out from the location to its neighbours
          nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
          for nbr_x, nbr_y in nbrs:
              fringe.append((nbr_x, nbr_y, dist+1))
      # no ghosts found
      return (10,-1)

    def getDirection(self, path):
      x1, y1 = path[0]
      x2, y2 = path[1]
      if x1 > x2:
          return 1
      elif x1 < x2:
          return 3
      elif y1 > y2:
          return 2
      else:
          return 0

    def closestFoodPath(self,pos, food, walls, ghosts, chemin_ban):
      """
      closestFood -- this is similar to the function that we have
      worked on in the search project; here its all in one place
      """
      fringe = [(pos[0], pos[1], 0, [])]
      expanded = set()
      for i in chemin_ban:
        expanded.add(i)
      while fringe:
          pos_x, pos_y, dist, path = fringe.pop(0)
          if (pos_x, pos_y) in expanded:
              continue
          expanded.add((pos_x, pos_y))
          # if we find a food at this location then exit
          if food[pos_x][pos_y]:
              path.append((pos_x, pos_y))
              return path

          if (pos_x,pos_y) in ghosts:
                continue

          # otherwise spread out from the location to its neighbours
          nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
          for nbr_x, nbr_y in nbrs:
              new_path = path.copy()
              new_path.append((pos_x, pos_y))
              fringe.append((nbr_x, nbr_y, dist+1, new_path))
      # no food found
      return None, []

    def getS5S10(self,food, walls, ghosts, pos_pacman, params, voisins, state):

        dist_max_menace, dist_max_ghost_mangeable = params

        tab_ghosts = []

        [tab_ghosts.append(self.closestGhost(pos_pacman, pos, ghosts, walls, state, dist_max_menace)) if pos is not None else tab_ghosts.append((0,-1)) for pos in voisins]

        # ça nous retourne la présence d'un fantôme d'une part, et si il est mangeable d'autre part.
        # on ajoute donc une vérification pour que la distance des fantômes mangeables soit 
        # celle donnée et non la même que celle des fantômes menaçants (même si je suis pas forcément 
        # convaincu de l'utilité de ne pas donner la même distance)

        distance_menace = [tab_ghosts[x][0] if tab_ghosts[x][1] == 0 else 10 for x in range(len(tab_ghosts))]

        if any(val < 3 for val in distance_menace):
            self.menace = 1
        else:
            self.menace = 0

        
        distance_mangeable = [tab_ghosts[x][0] if tab_ghosts[x][1] == 1 and tab_ghosts[x][0] <= dist_max_ghost_mangeable else 10 for x in range(len(tab_ghosts))]

        
        s5 = None
        s6s10 = []
        s10 = self.cherche_issue(pos_pacman, ghosts, walls, state, dist_max_menace)
        direction = voisins[:]
        
        # on check si un fantôme se trouve à proximité pour chacun des côtés
        for i in distance_menace:
            if i == 10:
                s6s10.append(0)
            else:
                s6s10.append(1)
        
        if s5 is None:
            boucle = 0
            # on fait attention aux fantômes menaçants
            nbNone = direction.count(None)
            while any(val < 10 for val in distance_menace) and boucle == 0:
                    if nbNone < 3:
                        idx = distance_menace.index(min(distance_menace))
                        direction[idx] = None
                        distance_menace[idx] = 11
                        nbNone += 1
                    else:
                        for i in range(4):
                            if direction[i] is not None:
                                s5 = i
                                boucle = 1
                                break
        
        if s5 is None:
            # si il n'y a pas de fantôme menaçant et qu'on ne cherche pas à fuir une menace, 
            # on regarde si il y a un fantôme mangeable
            if min(distance_mangeable) != 10:
                s5 = distance_mangeable.index(min(distance_mangeable))
            else:
                # si il n'y a pas de fantôme mangeable et pas de menace,
                # on prend la direction de la nourriture la + proche
                chemin_ban = []
                for i in range(4):
                    if distance_menace[i] == 11:
                        chemin_ban.append(voisins[i])

                path = self.closestFoodPath(pos_pacman, food, walls, ghosts, chemin_ban)
                if path != (None, []):
                    s5 = self.getDirection(path)
                else:
                    t = []
                    for i in range(len(direction)):
                        if direction[i] is not None:
                            t.append(i)
                    s5 = random.choice(t)
        s6s10.insert(0, s5)
        s6s10.append(s10)
        return s6s10

    def cherche_issue(self, pos_pacman, ghosts, walls, state, dist_max):
        fringe = [(pos_pacman[0], pos_pacman[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue

            if dist >= dist_max:
                return 0

            expanded.add((pos_x, pos_y))
            # if we find a ghosts at this location then exit
            if (pos_x,pos_y) in ghosts:
                continue
                
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # il n'existe pas de chemin d'une plus grande distance que 8
        # donc pacman est piégé
        return 1


class PacmanQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
    

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        featureVector = self.featExtractor.getFeatures(state, action)
        Q = self.weights * featureVector
        return Q

    def update(self, state, action, nextState, reward):
        featureVector = self.featExtractor.getFeatures(state, action)
        diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        for feature in featureVector:
            self.weights[feature] += self.alpha * diff * featureVector[feature]

    def computeValueFromQValues(self, state):
        if len(self.getLegalActions(state)) == 0:
            return 0.0
       
        #Q-Learning:
        max_value = self.getQValue(state, self.getLegalActions(state)[0])
        for action in self.getLegalActions(state)[1:] :
              Q = self.getQValue(state, action)
              if Q > max_value:
                    max_value = Q
        return max_value        
        
    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        
        if len(legalActions) == 0:
            return None
        start = random.randint(0, max(0,len(legalActions)-1))
        max_action = legalActions[start]
        max_action_value = self.getQValue(state, legalActions[start])
        for action in legalActions[0:]:
              Q = self.getQValue(state, action)
              if Q > max_action_value:
                    max_action_value = Q
                    max_action = action
        return max_action 

import random
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import tensorflow as tf
from memory_profiler import profile

"""class PacmanDEEPQAgent(QLearningAgent):
    def __init__(self, epsilon=1 , gamma=0.95, alpha=0.00025, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = 0.99 #gamma
        args['alpha'] = 0.000001    
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.list_action = ['North','West','South','East'] #,'Stop']
        self.nb_action = len(self.list_action)
        self.model = self.build_model()
        self.verif_model = self.build_model()
        self.compteur_verif = 0
        self.memory = deque(maxlen=10000)
        self.newstate = np.array([])
        self.batch_size = 64
        self.step = 0

    def build_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device available: ", device)
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.nb_action)
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr= self.alpha)
        loss_fn = nn.MSELoss()
        self.optimizer = optimizer
        return model

    def remember(self, state, action, nextState, reward, list_action):
        act = self.list_action.index(action)
        self.memory.append((state, act, reward, self.getNewStateCNN(nextState), self.getLegalActions(nextState), list_action))
          
    def replay(self, batch_size):
        self.model.train()
        minibatch = random.sample(self.memory, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        states = []
        targets = []
        for state, action, reward, nextState, nextLegalsActions, actions_possibles in minibatch:
            if len(nextLegalsActions) == 0:
                target = reward
            else:
                with torch.no_grad():
                    next_state_values = self.model(nextState.to(device))
                
                maxi = -10
                for i in range(len(next_state_values[0])):
                    if self.list_action[i] in nextLegalsActions:
                        if next_state_values[0][i] > maxi:
                            maxi = next_state_values[0][i]

                target = reward + self.discount * maxi

            target_f = self.model(state.to(device))

            for i in range(len(target_f[0])):
                    if self.list_action[i] not in actions_possibles:
                        target_f[0][i] = -3

            target_f[0][action] = target

            states.append(state)
            targets.append(target_f)
            
        states = torch.cat(states).to(device)
        targets = torch.cat(targets).to(device)
        loss_fn = nn.MSELoss()
        loss = loss_fn(targets, targets.view(-1, self.nb_action))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, state, action, nextState, reward):
        self.remember(self.newstate, action, nextState, reward, self.getLegalActions(state))
        self.replay(min(self.batch_size, len(self.memory)))
        if self.epsilon > 0.001:
            self.epsilon = self.epsilon * 0.9999998 

    def getAction(self, state):
        self.step += 1
        self.model.eval()
        self.newstate = self.getNewStateCNN(state)
        action = None
        if util.flipCoin(self.epsilon):
            while action not in self.list_action:
                action = random.choice(self.getLegalActions(state))
        else:

            #MA VERSION
            with torch.no_grad():
                prob_action = self.model(self.newstate.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().numpy()[0] #self.getPolicy(state)
            while action is None:
                max_index = np.argmax(prob_action)
                if self.list_action[max_index] in self.getLegalActions(state):
                    action = self.list_action[max_index]
                else:
                    prob_action[max_index] = -100
        self.doAction(state, action)
        return action

    def getNewStateCNN(self, state):
        statetab = [list(s) for s in state.__str__().strip().splitlines()]
        colors = {'%': (0, 0, 255), 'G': (255, 0, 0), '>': (200, 200, 0), '<': (200, 200, 0), '^': (200, 200, 0), 'v': (200, 200, 0), 'P': (200, 200, 0), '.': (255, 255, 255)}
        image = Image.new("RGB", (len(statetab[0]), len(statetab)), (0, 0, 0))
        #print([list(s) for s in state.__str__().strip().splitlines()])
        aaa =  image.save(f"image/{self.step}.jpg")
        for i in range(len(statetab)):
            for j in range(len(statetab[0])):
                pixel_color = colors.get(statetab[i][j], (0, 0, 0))
                image.putpixel((j, i), pixel_color)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.tensor(img_array).unsqueeze(0)
        return img_tensor
"""


class PacmanDEEPQAgent(QLearningAgent):
    def __init__(self, epsilon=1 , gamma=0.95, alpha=0.00025, numTraining=0, **args):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.list_action = ['North','West','South','East'] #,'Stop']
        self.nb_action = len(self.list_action)
        self.memory = []
        self.newstate = np.array([])
        self.batch_size = 16
        self.model = self.build_model_keras((7,7,3), len(self.list_action))
        self.step = 0

    def build_model_keras(self, input_shape, num_actions):
        model = Sequential()
        model.add(Conv2D(16, (4, 4), strides=(1, 1), activation='tanh', input_shape=input_shape))  # relu -> tanh
        model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='tanh'))
        model.add(Flatten()) #512
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def getAction(self, state):
        self.newstate = self.getNewStateCNN(state)
        if len(self.getLegalActions(state)) == 1:
            return self.getLegalActions(state)[0]
        action = None
        if util.flipCoin(self.epsilon):
            t = self.getLegalActions(state)
            t.remove("Stop")
            action = random.choice(t)
        else:
            prob_action = self.model.predict(self.newstate, verbose = 0)[0]
            #print(prob_action)
            while action is None:
                max_index = np.argmax(prob_action)
                if self.list_action[max_index] in self.getLegalActions(state):
                    action = self.list_action[max_index]
                else:
                    prob_action[max_index] = - 500
            del prob_action
        self.doAction(state, action)
        self.step += 1
        return action
    
    def replay(self, batch_size):
        #print(f"replay {self.step}")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, legal in minibatch:
            pred = self.model.predict(next_state, verbose = 0)[0]
            target = reward + self.gamma * np.amax(pred)
            target_f = self.model.predict(state, verbose = 0)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
            del pred
            del target_f
        tf.keras.backend.clear_session()
        del minibatch

    
    def update(self, state, action, nextState, reward): 
        self.remember(action, nextState, reward, self.getLegalActions(state))
        if self.step % 5 == 0:
            self.replay(min(self.batch_size, len(self.memory)))

        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * 0.999 

    def remember(self, action, nextState, reward, list_action):
        act = self.list_action.index(action)
        if len(self.memory) >= 1000 :
            self.memory.pop(0)
        self.memory.append((self.newstate, act, reward, self.getNewStateCNN(nextState), self.getLegalActions(nextState)))
        if len(self.memory) % 99 == 0:
            print(len(self.memory))

    def getNewStateCNN(self, state):
        statetab = [list(s) for s in state.__str__().strip().splitlines()][:-1]
        colors = {'%': (0, 0, 255), 'G': (255, 0, 0), '>': (200, 200, 0), '<': (200, 200, 0), '^': (200, 200, 0), 'v': (200, 200, 0), 'P': (200, 200, 0), '.': (255, 255, 255)}
        image = Image.new("RGB", (len(statetab[0]), len(statetab)), (0, 0, 0))
        
        for i in range(len(statetab)):
            for j in range(len(statetab[0])):
                pixel_color = colors.get(statetab[i][j], (0, 0, 0))
                image.putpixel((j, i), pixel_color)
                
        #s =  image.save(f"image/{self.step}.png")
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # mémoire
        del image
        return img_array
    

"""class PacmanDEEPQAgent(QLearningAgent):
    def __init__(self, epsilon=1 , gamma=0.95, alpha=0.00025, numTraining=0, **args):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.list_action = ['North','West','South','East'] #,'Stop']
        self.nb_action = len(self.list_action)
        self.memory = []
        self.newstate = np.array([])
        self.batch_size = 16
        self.model = self.build_model_keras(len(self.list_action))
        self.step = 0

    def build_model_keras(self, num_actions):
        model = Sequential()
        model.add(Dense(64, activation='tanh', input_shape= (50,)))
        model.add(Dense(128, activation='tanh', ))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def getAction(self, state):
        self.newstate = self.getNewStateCNN(state)
        if len(self.getLegalActions(state)) == 1:
            return self.getLegalActions(state)[0]
        action = None
        if util.flipCoin(self.epsilon):
            t = self.getLegalActions(state)
            t.remove("Stop")
            action = random.choice(t)
        else:
            prob_action = self.model.predict(self.newstate, verbose = 0)[0]
            #print(prob_action)
            while action is None:
                max_index = np.argmax(prob_action)
                if self.list_action[max_index] in self.getLegalActions(state):
                    action = self.list_action[max_index]
                else:
                    prob_action[max_index] = - 500
            del prob_action
        self.doAction(state, action)
        self.step += 1
        return action
    
    def replay(self, batch_size):
        #print(f"replay {self.step}")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, legal in minibatch:
            pred = self.model.predict(next_state, verbose = 0)[0]
            target = reward + self.gamma * np.amax(pred)
            target_f = self.model.predict(state, verbose = 0)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
            del pred
            del target_f
        tf.keras.backend.clear_session()
        del minibatch

    
    def update(self, state, action, nextState, reward): 
        self.remember(action, nextState, reward, self.getLegalActions(state))
        if self.step % 5 == 0:
            self.replay(min(self.batch_size, len(self.memory)))

        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * 0.999998 

    def remember(self, action, nextState, reward, list_action):
        act = self.list_action.index(action)
        if len(self.memory) >= 1000 :
            self.memory.pop(0)
        self.memory.append((self.newstate, act, reward, self.getNewStateCNN(nextState), self.getLegalActions(nextState)))
        if len(self.memory) % 99 == 0:
            print(len(self.memory))

    def getNewStateCNN(self, state):
        statetab = [list(s) for s in state.__str__().strip().splitlines()][:-1]
        state_clean = [sublist[1:-1] for sublist in statetab[1:-1]] 
        P_G = np.array([])
        tab_gum = np.array([])
        gum_dico = {'%': -1, '.': 1, ' ' : 0, 'G': 0, '>': 0, '<': 0, '^': 0, 'v': 0, 'P': 0}
        P_G_dico = {'G': -1, '>': 1, '<': 1, '^': 1, 'v': 1, 'P': 1, '%': 0, '.': 0, ' ' : 0}

        for i in range(len(state_clean)):
            for j in range(len(state_clean[0])):
                P_G = np.append(P_G, P_G_dico.get(state_clean[i][j]))
                tab_gum = np.append(tab_gum, gum_dico.get(state_clean[i][j]))

        out = np.concatenate((P_G, tab_gum), axis = 0)
        out = np.reshape(out, (1, 50))
        del state_clean
        del statetab
        del P_G
        del gum_dico
        del tab_gum
        del P_G_dico

        return out"""