# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        ghostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodweight = 9.0
        ghostweight = 9.0
        fooddistance = []

        value = successorGameState.getScore()

        distanceToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distanceToGhost > 0:
            value -= ghostweight / distanceToGhost

        for i in newFood.asList():
            fooddistance.append(manhattanDistance(newPos,i))


        if len(fooddistance):
            value += foodweight / min(fooddistance)
        return value
        #return successorGameState.getScore()

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[0]

    ## we will check if we have reached the maximum depth limit, or we have lost or won
    ## afterwards we will check if its the turn of pacman or the ghost
    ##if its the turn of the ghost we will call the min function otherwise if its the turn of pacman we call the max function 
    ##the value function is called recursively in the min and max value functions
    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minval(gameState, depth)
        else:
            return self.maxval(gameState, depth)

    def minval(self, gameState, depth):
        ##we get the legal actions for the agent whose turn it is. The agent index depends on the level of depth we are at 
        actions_allowed = gameState.getLegalActions(depth % gameState.getNumAgents())
        ##we initialize the following to positive infinity, and also store the best action to get the minimum value
        minimum_result = ("", float("Inf"))

        if len(actions_allowed) == 0:
            return "", self.evaluationFunction(gameState)

        for move in actions_allowed:
            child = gameState.generateSuccessor(depth % gameState.getNumAgents(), move)
            result = self.value(child, depth + 1)
            if result[1] < minimum_result[1]:
                minimum_result = (move, result[1])
        return minimum_result ## we return the action and the minimum value we can get 

    ##core concept stays the same. However, we just find the max value now 
    def maxval(self, gameState, depth):
        actions_allowed = gameState.getLegalActions(0)
        maximum_result = ("", float("-Inf"))

        if len(actions_allowed) == 0:
            return "", self.evaluationFunction(gameState)

        for move in actions_allowed:
            child = gameState.generateSuccessor(0, move)
            result = self.value(child, depth + 1)
            if result[1] > maximum_result[1]:
                maximum_result = (move, result[1])
        return maximum_result
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ##we just add alpha and beta values to our code and initialize them to -infinity and +infinity
        return self.value(gameState, 0, float("-Inf"), float("Inf"))[0]


    def value(self, gameState, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.minval(gameState, depth, alpha, beta)
        else:
            return self.maxval(gameState, depth, alpha, beta)

    def minval(self, gameState, depth, alpha, beta):
        ##we get the legal actions for the agent whose turn it is. The agent index depends on the level of depth we are at 
        actions_allowed = gameState.getLegalActions(depth % gameState.getNumAgents())
        ##we initialize the following to positive infinity, and also store the best action to get the minimum value
        minimum_result = ("", float("Inf"))

        if len(actions_allowed) == 0:
            return "", self.evaluationFunction(gameState)

        for move in actions_allowed:
            child = gameState.generateSuccessor(depth % gameState.getNumAgents(), move)
            result = self.value(child, depth + 1, alpha, beta)
            if result[1] < minimum_result[1]:
                minimum_result = (move, result[1])
            if minimum_result[1] < alpha:
                return minimum_result 
            beta = min(beta, minimum_result[1])
        return minimum_result ## we return the action and the minimum value we can get 

    ##core concept stays the same. However, we just find the max value now 
    def maxval(self, gameState, depth, alpha, beta):
        actions_allowed = gameState.getLegalActions(0)
        maximum_result = ("", float("-Inf"))

        if len(actions_allowed) == 0:
            return "", self.evaluationFunction(gameState)

        for move in actions_allowed:
            child = gameState.generateSuccessor(0, move)
            result = self.value(child, depth + 1,alpha,beta)
            if result[1] > maximum_result[1]:
                maximum_result = (move, result[1])
            if maximum_result[1] > beta:
                return maximum_result
            alpha = max(alpha, maximum_result[1])
        return maximum_result
        util.raiseNotDefined()
        util.raiseNotDefined()

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
        return self.value(gameState, 0)[0]

    ##value function and the maxval function will remain the same as our agent(pacman) is still the maximizer. 
    ##However, only the minval function from our minimax class will change now 

    def value(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return "", self.evaluationFunction(gameState)
        if depth % gameState.getNumAgents() != 0:
            return self.expectedval(gameState, depth)
        else:
            return self.maxval(gameState, depth)

    def expectedval(self, gameState, depth):
        actions_allowed = gameState.getLegalActions(depth % gameState.getNumAgents())
        expectedvalue = 0
        prob_weight = 1. / len(actions_allowed)

        if len(actions_allowed) == 0:
            return ("", self.evaluationFunction(gameState))


        for move in actions_allowed:
            child = gameState.generateSuccessor(depth % gameState.getNumAgents(), move)
            result = self.value(child, depth + 1)
            expectedvalue = expectedvalue + (result[1]*prob_weight)
        return ("",expectedvalue)

    ##core concept stays the same. However, we just find the max value now 
    def maxval(self, gameState, depth):
        actions_allowed = gameState.getLegalActions(0)
        maximum_result = ("", float("-Inf"))

        if len(actions_allowed) == 0:
            return ("", self.evaluationFunction(gameState))

        for move in actions_allowed:
            child = gameState.generateSuccessor(0, move)
            result = self.value(child, depth + 1)
            if result[1] > maximum_result[1]:
                maximum_result = (move, result[1])
        return maximum_result
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Generally my evaluation function considers two cases when the ghosts are scared and when the ghosts are not scared. When the ghosts are not scared 
    I consider the getscore() plus reciprocal of the distance to the food, however when the ghosts are scared then it is good if pacman is closer to
    the good as pacman can also eat them so I add this to the previous values mentioned. 
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodlist = newFood.asList()
    #total_power_pellets = len(currentGameState.getCapsules())
    ghost_distance = []
    food_distance = []
    fooddistreciprocal = 0
    total_scared_time = sum(newScaredTimes)
    #nofoodsum = len(newFood.asList(False))
    score = 0

    ##find distance to each ghost 
    for ghost in newGhostStates:
        x = ghost.getPosition()
        ghost_distance.append(manhattanDistance(newPos,x))

    total_ghost_distance = sum(ghost_distance)

    ##find distance to food 
    for food in foodlist:
        food_distance.append(manhattanDistance(newPos,food))

    val = sum(food_distance)
    if val > 0:
        fooddistreciprocal = 1.0 / val


    if total_scared_time > 0:
        score = score + total_scared_time + (-0.5 * total_ghost_distance) + currentGameState.getScore() + fooddistreciprocal 
    else: 
        score = score + total_ghost_distance + currentGameState.getScore() + fooddistreciprocal 
    return score 




    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
