# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from cmath import inf
from itertools import accumulate
from queue import PriorityQueue
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    # Initialize the frontier using a stack (LIFO)
    frontier = util.Stack()
    startState = problem.getStartState()
    frontier.push((startState, []))

    # Set to keep track of visited states
    visited = set()

    while not frontier.isEmpty():
        currentState, actions = frontier.pop()

        # If the current state is the goal state, return the actions
        if problem.isGoalState(currentState):
            return actions

        # Mark the current state as visited
        visited.add(currentState)

        # Get successors of the current state
        successors = problem.getSuccessors(currentState)

        for nextState, nextAction, _ in successors:
            if nextState not in visited:
                newActions = actions + [nextAction]
                frontier.push((nextState, newActions))

    # If no solution was found, raise an exception
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Initialize the frontier using a queue (FIFO)
    frontier = util.Queue()
    startState = problem.getStartState()
    frontier.push((startState, []))

    # Set to keep track of visited states
    visited = set()

    while not frontier.isEmpty():
        currentState, actions = frontier.pop()

        # If the current state is the goal state, return the actions
        if problem.isGoalState(currentState):
            return actions

        # Mark the current state as visited
        visited.add(currentState)

        # Get successors of the current state
        successors = problem.getSuccessors(currentState)

        for nextState, nextAction, _ in successors:
            if nextState not in visited:
                newActions = actions + [nextAction]
                frontier.push((nextState, newActions))

    # If no solution was found, raise an exception
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # Initialize the frontier using a priority queue
    frontier = util.PriorityQueue()
    startState = problem.getStartState()
    frontier.push((startState, []), 0)

    # Dictionary to keep track of visited states and their costs
    visited = {}

    while not frontier.isEmpty():
        currentState, actions = frontier.pop()

        # If the current state is the goal state, return the actions
        if problem.isGoalState(currentState):
            return actions

        # If the state is not visited or the new path to the state is cheaper
        if currentState not in visited or visited[currentState] > problem.getCostOfActions(actions):
            visited[currentState] = problem.getCostOfActions(actions)

            # Get successors of the current state
            successors = problem.getSuccessors(currentState)

            for nextState, nextAction, stepCost in successors:
                newActions = actions + [nextAction]
                newCost = problem.getCostOfActions(newActions)
                if nextState not in visited or visited[nextState] > newCost:
                    frontier.push((nextState, newActions), newCost)

    # If no solution was found, raise an exception
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Please DO NOT change the following code, we will use it later
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '',0, [])
    myPQ.push(startNode,heuristic(startState,problem))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, action, cost, path = node
        if (not state in visited) or cost < best_g.get(state):
            visited.add(state)
            best_g[state]=cost
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                myPQ.push(newNode,heuristic(succState,problem)+cost+succCost)
    util.raiseNotDefined()


def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """

    # Procedure to improve the current state using BFS until a better heuristic value is found
    def improve(initialNode):
        queue = util.Queue()  # FIFO queue for BFS
        queue.push(initialNode)
        closed = set()

        while not queue.isEmpty():
            currentNode = queue.pop()
            currentState = currentNode[0]
            currentActions = currentNode[1]

            if currentState not in closed:
                closed.add(currentState)

                # If the heuristic of the current state is better than the initial state, return the node
                if heuristic(currentState, problem) < heuristic(initialNode[0], problem):
                    return currentNode

                for successor, action, _ in problem.getSuccessors(currentState):
                    queue.push((successor, currentActions + [action]))

        return None  # No better state found

    # Start the enforced hill-climbing
    startState = problem.getStartState()
    currentNode = (startState, [])

    while not problem.isGoalState(currentNode[0]):
        improvedNode = improve(currentNode)
        if improvedNode:
            currentNode = improvedNode
        else:
            # If no better state is found, the search is stuck and should be terminated
            return []

    return currentNode[1]  # Return the actions leading to the goal state




from math import inf as INF
from util import PriorityQueue

def bidirectionalAStarEnhanced(problem, heuristic=nullHeuristic, backwardsHeuristic=nullHeuristic):
    """Bidirectional global search with heuristic function."""

    # Initialize the priority queues for forward and backward search
    Open_f = PriorityQueue()
    Open_b = PriorityQueue()
    Closed_f = {}
    Closed_b = {}

    # Start state and goal state
    startState = problem.getStartState()
    goalState = problem.goal

    # Push the start and goal states into the respective priority queues
    Open_f.push(startState, heuristic(startState, problem))
    Open_b.push(goalState, backwardsHeuristic(goalState, problem))

    Closed_f[startState] = (None, None, 0)  # Initialize the start state with g-value of 0
    Closed_b[goalState] = (None, None, 0)   # Initialize the goal state with g-value of 0

    while not Open_f.isEmpty() and not Open_b.isEmpty():
        # Forward search
        state_f = Open_f.pop()
        if state_f in Closed_b:
            # Path found
            path_f = reconstruct_path(Closed_f, state_f)
            path_b = reconstruct_path(Closed_b, state_f)[::-1]
            return path_f + path_b

        for successor, action, cost in problem.getSuccessors(state_f):
            g_value_f = Closed_f[state_f][2] + cost
            if successor not in Closed_f or g_value_f < Closed_f[successor][2]:
                f_value_f = g_value_f + heuristic(successor, problem)
                Open_f.push(successor, f_value_f)
                Closed_f[successor] = (state_f, action, g_value_f)

        # Backward search
        state_b = Open_b.pop()
        if state_b in Closed_f:
            # Path found
            path_f = reconstruct_path(Closed_f, state_b)
            path_b = reconstruct_path(Closed_b, state_b)[::-1]
            return path_f + path_b

        for successor, action, cost in problem.getBackwardsSuccessors(state_b):
            g_value_b = Closed_b[state_b][2] + cost
            if successor not in Closed_b or g_value_b < Closed_b[successor][2]:
                f_value_b = g_value_b + backwardsHeuristic(successor, problem)
                Open_b.push(successor, f_value_b)
                Closed_b[successor] = (state_b, action, g_value_b)

    # If no path found
    return []

def reconstruct_path(closed_list, state):
    """Helper function to reconstruct the path from the start state to the given state."""
    path = []
    while state in closed_list and closed_list[state][1] is not None:
        _, action, _ = closed_list[state]
        path.append(action)
        state = closed_list[state][0]
    return path[::-1]






# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


ehc = enforcedHillClimbing
bae = bidirectionalAStarEnhanced


