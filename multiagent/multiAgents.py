# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Directions
from util import manhattanDistance
import random, util, searchAgents

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    
    def distCloseFood(position, foodGrid) :
        gridInfo = foodGrid.packBits()
        
        value = None
        for i in range(gridInfo[0]) :
            for j in range(gridInfo[1]) :
                if foodGrid[i][j] == True :
                    dist = searchAgents.mazeDistance(position, (i,j), successorGameState)
                    #dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                    if value == None :
                        value = dist
                    if dist < value :
                        value = dist
        if value == None : 
            value = (0, position)
        return value
    
    def distClose(position, foodGrid) :
        gridInfo = foodGrid.packBits()
        
        value = None
        for i in range(gridInfo[0]) :
            for j in range(gridInfo[1]) :
                if foodGrid[i][j] == True :
                    dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                    if value == None :
                        value = dist
                    if dist[0] < value[0] :
                        value = dist
        if value == None : 
            value = (0, position)
        return value
    
    
    
#    print '----------------------EVALUATION------------------------'
#    print 'currentGameState:', currentGameState
#    print 'successorGameState:', successorGameState
#    print 'newPos:', newPos
#    print 'newFood:', newFood
#    print 'newFood.count():', newFood.count()
#    print 'newGhostStates:', newGhostStates
#    print 'newGhostStates[0]:', newGhostStates[0]
#    print 'newGhostStates[0].getPosition():', newGhostStates[0].getPosition()
#    print 'successorGameState.getLegalActions()', successorGameState.getLegalActions()
#    print 'newScaredTimes:', newScaredTimes[0]
#    print 'ghost locs:',
#    print (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[0])
#    fp = searchAgents.FoodSearchProblem(successorGameState)
#    print 'searchAgents.foodHeuristic((newPos,newFood),fp )', searchAgents.foodHeuristic((newPos,newFood),fp )
#    
    
    #score = 1/newFood.count()
    score = 0
    score= successorGameState.getScore()
    if newFood.count() > 0 :
        score += 1/newFood.count() * 100
    else :
#        print 'returnEval', score
        return 1000
    if newScaredTimes[0] < 1 :
        if newPos == (newGhostStates[0].getPosition()[0]-1, newGhostStates[0].getPosition()[1]):
            score = -10000
        elif newPos == (newGhostStates[0].getPosition()[0]+1, newGhostStates[0].getPosition()[1]):
            score = -10000
        elif newPos == (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]-1):
            score = -10000
        elif newPos == (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]+1):
            score = -10000
    

#    i = int(6.0)
#    ghostPos = ( int(newGhostStates[0].getPosition()[0]), int(newGhostStates[0].getPosition()[1]) )
#    score -= distCloseFood(newPos,newFood)
    if newFood.count() == currentGameState.getFood().count() :
        score -= distClose(newPos,newFood)[0]
        
    score += newScaredTimes[0] * 10
        
#    if ghostPos < 2 :
#        return -10
#    score += 1/2 * searchAgents.mazeDistance(newPos,ghostPos,currentGameState )
    
    
    
#    print 'returnEval', score
    return score 

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def minMax(state, checkD = -1, agentsTurn = -1):
        agentsTurn += 1
        agentsTurn = agentsTurn % gameState.getNumAgents()
        if agentsTurn == 0 :
            checkD += 1
        if state.isWin() or state.isLose() or (checkD == self.depth) :
            return self.evaluationFunction(state)
        if agentsTurn == 0 :
            v = maxValue(state, checkD, agentsTurn)
            return v
        else :
            v = minValue(state, checkD, agentsTurn)
            return v
        
    def maxValue(state, checkD, agentsTurn):
        max = float("-inf")
        actions = state.getLegalActions(agentsTurn)
        if checkD == 0:
            max = (max, 'Stop')
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = ( minMax(successor, checkD, agentsTurn), a)
                if v[0] > max[0] :
                    max = v
        else : 
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = minMax(successor, checkD, agentsTurn)
                if v > max :
                    max = v
        return max
    
    def minValue(state, checkD, agentsTurn):
        min = float("inf")
        actions = state.getLegalActions(agentsTurn) 
        for a in actions :
            successor = state.generateSuccessor(agentsTurn,a)
            v = minMax(successor, checkD, agentsTurn)
            if v < min :
                min = v
        return min

    action = minMax(gameState)
    return action[1]

alpha = 0
beta = 0       
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    global alpha
    global beta
    alpha = float("-inf")
    beta = float("inf")  
    
    def minMax(state, checkD, agentsTurn, alpha, beta):
        agentsTurn += 1
        agentsTurn = agentsTurn % gameState.getNumAgents()
        if agentsTurn == 0 :
            checkD += 1
        if state.isWin() or state.isLose() or (checkD == self.depth) :
            return self.evaluationFunction(state)
        if agentsTurn == 0 :
            v = maxValue(state, checkD, agentsTurn, alpha, beta)
            return v
        else :
            v = minValue(state, checkD, agentsTurn, alpha, beta)
            return v
        
    def maxValue(state, checkD, agentsTurn, alpha, beta):
        maxVal = float("-inf")
        actions = state.getLegalActions(agentsTurn)
        if checkD == 0:
            maxVal = (maxVal, 'Stop')
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = ( minMax(successor, checkD, agentsTurn, alpha, beta), a)
                if v[0] >= beta :
                    return v
                if v[0] > maxVal[0] :
                    maxVal = v
                alpha = max(alpha, v[0])
        else : 
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = minMax(successor, checkD, agentsTurn, alpha, beta)
                if v >= beta :
                    return v
                if v > maxVal :
                    maxVal = v
                alpha = max(alpha, v)
        return maxVal
    
    def minValue(state, checkD, agentsTurn, alpha, beta):
        minVal = float("inf")
        actions = state.getLegalActions(agentsTurn) 
        for a in actions :
            successor = state.generateSuccessor(agentsTurn,a)
            v = minMax(successor, checkD, agentsTurn, alpha, beta)
            if v <= alpha :
                return v
            if v < minVal :
                minVal = v
            beta = min(beta, v)
        return minVal
    
    action = minMax(gameState, -1, -1, alpha, beta)
    return action[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def minMax(state, checkD = -1, agentsTurn = -1):
        agentsTurn += 1
        agentsTurn = agentsTurn % gameState.getNumAgents()
        if agentsTurn == 0 :
            checkD += 1
        if state.isWin() or state.isLose() or (checkD == self.depth) :
            return self.evaluationFunction(state)
        if agentsTurn == 0 :
            v = maxValue(state, checkD, agentsTurn)
            return v
        else :
            v = minValue(state, checkD, agentsTurn)
            return v
        
    def maxValue(state, checkD, agentsTurn):
        max = float("-inf")
        actions = state.getLegalActions(agentsTurn)
        if checkD == 0:
            max = (max, 'Stop')
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = ( minMax(successor, checkD, agentsTurn), a)
                if v[0] > max[0] :
                    max = v
        else : 
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = minMax(successor, checkD, agentsTurn)
                if v > max :
                    max = v
        return max
    
    def minValue(state, checkD, agentsTurn):
        min = float(0)
        actions = state.getLegalActions(agentsTurn) 
        for a in actions :
            successor = state.generateSuccessor(agentsTurn,a)
            v = minMax(successor, checkD, agentsTurn)
            min += v/len(actions)
        return min

    action = minMax(gameState)
    return action[1]


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  
  # just initizilizing some varibles here
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  "*** YOUR CODE HERE ***"
  # returns the distance to the closest food (maze distance)
  def distCloseFood(position, foodGrid) :
      gridInfo = foodGrid.packBits()
      
      value = None
      for i in range(gridInfo[0]) :
          for j in range(gridInfo[1]) :
              if foodGrid[i][j] == True :
                  dist = searchAgents.mazeDistance(position, (i,j), currentGameState)
                  #dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                  if value == None :
                      value = dist
                  if dist < value :
                      value = dist
      if value == None : 
          value = (0, position)
      return value
  # returns the distance to the closest food (manhatten distance)
  def distClose(position, foodGrid) :
      gridInfo = foodGrid.packBits()
      
      value = None
      for i in range(gridInfo[0]) :
          for j in range(gridInfo[1]) :
              if foodGrid[i][j] == True :
                  dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                  if value == None :
                      value = dist
                  if dist[0] < value[0] :
                      value = dist
      if value == None : 
          value = (0, position)
      return value
  



#  print '----------------------EVALUATION------------------------'
#  print 'currentGameState:', currentGameState
#  print 'currentGameState:', currentGameState
#  print 'newPos:', newPos
#  print 'newFood:', newFood
#  print 'newFood.count():', newFood.count()
#  print 'newGhostStates:', newGhostStates
#  print 'newGhostStates[0]:', newGhostStates[0]
#  print 'newGhostStates[0].getPosition():', newGhostStates[0].getPosition()
#  print 'currentGameState.getLegalActions()', currentGameState.getLegalActions()
#  print 'newScaredTimes:', newScaredTimes[0]
#  print 'ghost locs:',
#  print (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[0])
#  fp = searchAgents.FoodSearchProblem(currentGameState)
#  print 'searchAgents.foodHeuristic((newPos,newFood),fp )', searchAgents.foodHeuristic((newPos,newFood),fp )
    
# # worked with a submit
 
#  #score = 1/newFood.count()
#  score = 0
#  
#  # just addes the game score
#  score= currentGameState.getScore()
#  
#  # addes  1/food-count multiplied by 100 to the score if there is food left, if not returns 1000 for a win
#  if newFood.count() > 0 :
#      score += 1/newFood.count() * 100
#  else :
##        print 'returnEval', score
#      return 1000
#  
#  # if the ghost isn't scared run away 
#  if newScaredTimes[0] < 1 :
#      if newPos == (newGhostStates[0].getPosition()[0]-1, newGhostStates[0].getPosition()[1]):
#          score = -10000
#      elif newPos == (newGhostStates[0].getPosition()[0]+1, newGhostStates[0].getPosition()[1]):
#          score = -10000
#      elif newPos == (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]-1):
#          score = -10000
#      elif newPos == (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]+1):
#          score = -10000
#  
#
#
#  # always true... just subtract the distance to the closest food
#  if newFood.count() == currentGameState.getFood().count() :
#      score -= distClose(newPos,newFood)[0]
#      
#  # adds for scared times, making him get capsules
#  score += newScaredTimes[0] * 10
  
#    print 'returnEval', score


  #score = 1/newFood.count()
  score = 0
  
  # just addes the game score
  score= currentGameState.getScore()
  
  if currentGameState.isWin() :
      return 1000 + score 
  
  # addes  1/food-count multiplied by 100 to the score if there is food left, if not returns 1000 for a win
  if newFood.count() > 0 :
      score += ( (1/newFood.count() )* 200 )
  
  # if the ghost isn't scared run away, if it is go get em
  for i in range(len(newGhostStates)) :
      ghostPos = newGhostStates[i].getPosition()
      if newScaredTimes[i] < 1 :
          if newPos == (ghostPos[0]-1, ghostPos[1]):
              score = -10000
          elif newPos == (ghostPos[0]+1, ghostPos[1]):
              score = -10000
          elif newPos == (ghostPos[0], ghostPos[1]-1):
              score = -10000
          elif newPos == (ghostPos[0], ghostPos[1]+1):
              score = -10000
      else :
          score += ( (1/( abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])) ) * 100 )



  #subtract the distance to the closest food
  score -= distClose(newPos,newFood)[0]
      
  # adds for scared times, making him get capsules
#  for i in range(len(newGhostStates)) :
#      score += newScaredTimes[i] * 10
  

  return score 

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    # just initizilizing some varibles here
    
    global alpha
    global beta
    alpha = float("-inf")
    beta = float("inf")  
    
    def evalFunction(state): 
        newPos = state.getPacmanPosition()
        newFood = state.getFood()
        newGhostStates = state.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        def distCloseFood(position, foodGrid) :
            gridInfo = foodGrid.packBits()
            
            value = None
            for i in range(gridInfo[0]) :
                for j in range(gridInfo[1]) :
                    if foodGrid[i][j] == True :
                        dist = searchAgents.mazeDistance(position, (i,j), state)
                        #dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                        if value == None :
                            value = dist
                        if dist < value :
                            value = dist
            if value == None : 
                value = (0, position)
            return value
          # returns the distance to the closest food (manhatten distance)
        def distClose(position, foodGrid) :
            gridInfo = foodGrid.packBits()
            
            value = None
            for i in range(gridInfo[0]) :
                for j in range(gridInfo[1]) :
                    if foodGrid[i][j] == True :
                        dist = ( ( abs(position[0] - i) + abs(position[1] - j) ), (i,j) )
                        if value == None :
                            value = dist
                        if dist[0] < value[0] :
                            value = dist
            if value == None : 
                value = (0, position)
            return value
        
        
        score = 0
  
        # just addes the game score
        score= state.getScore()
        
        if state.isWin() :
            return 1000 + score 
        
        # addes  1/food-count multiplied by 100 to the score if there is food left, if not returns 1000 for a win
        if newFood.count() > 0 :
            score += ( (1/newFood.count() )* 200 )
        
        # if the ghost isn't scared run away, if it is go get em
        for i in range(len(newGhostStates)) :
            ghostPos = newGhostStates[i].getPosition()
            if newScaredTimes[i] < 1 :
                if newPos == (ghostPos[0]-1, ghostPos[1]):
                    score = -10000
                elif newPos == (ghostPos[0]+1, ghostPos[1]):
                    score = -10000
                elif newPos == (ghostPos[0], ghostPos[1]-1):
                    score = -10000
                elif newPos == (ghostPos[0], ghostPos[1]+1):
                    score = -10000
            else :
                score += ( (1/( abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])) ) * 100 )
        
        
        
        #subtract the distance to the closest food
        score -= distClose(newPos,newFood)[0]
        
        return score
    
    def minMax(state, checkD, agentsTurn, alpha, beta):
        agentsTurn += 1
        agentsTurn = agentsTurn % state.getNumAgents()
        if agentsTurn == 0 :
            checkD += 1
        if state.isWin() or state.isLose() or (checkD == 3) :
            return evalFunction(state)
        if agentsTurn == 0 :
            v = maxValue(state, checkD, agentsTurn, alpha, beta)
            return v
        else :
            v = minValue(state, checkD, agentsTurn, alpha, beta)
            return v
    
    def maxValue(state, checkD, agentsTurn, alpha, beta):
        maxVal = float("-inf")
        actions = state.getLegalActions(agentsTurn)
        if checkD == 0:
            maxVal = (maxVal, 'Stop')
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = ( minMax(successor, checkD, agentsTurn, alpha, beta), a)
                if v[0] >= beta :
                    return v
                if v[0] > maxVal[0] :
                    maxVal = v
                alpha = max(alpha, v[0])
        else : 
            for a in actions :
                successor = state.generateSuccessor(agentsTurn,a)
                v = minMax(successor, checkD, agentsTurn, alpha, beta)
                if v >= beta :
                    return v
                if v > maxVal :
                    maxVal = v
                alpha = max(alpha, v)
        return maxVal
    
    def minValue(state, checkD, agentsTurn, alpha, beta):
        minVal = float("inf")
        actions = state.getLegalActions(agentsTurn) 
        for a in actions :
            successor = state.generateSuccessor(agentsTurn,a)
            v = minMax(successor, checkD, agentsTurn, alpha, beta)
            if v <= alpha :
                return v
            if v < minVal :
                minVal = v
            beta = min(beta, v)
        return minVal
    
    action = minMax(gameState, -1, -1, alpha, beta)
    return action[1]
     