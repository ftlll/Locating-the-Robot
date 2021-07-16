# version 1.0

import numpy as np
from typing import List, Dict

from utils_soln import *

def create_observation_matrix(env: Environment):
    '''
    Creates a 2D numpy array containing the observation probabilities for each state. 

    Entry (i,j) in the array is the probability of making an observation type j in state i.

    Saves the matrix in env.observe_matrix and returns nothing.
    '''

    #### Your Code Here ####
    pass


def create_transition_matrices(env: Environment):
    '''
    If the transition_matrices in env is not None, 
    constructs a 3D numpy array containing the transition matrix for each action.

    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.

    Saves the matrices in env.transition_matrices and returns nothing.
    '''

    if env.transition_matrices is not None:
        return

    #### Your Code Here ####
    pass


def forward_recursion(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform the filtering task for all the time steps.

    Calculate and return the values f_{0:0} to f_{0,t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.
    :param probs_init: The initial probabilities over the N states.

    :return: A numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the normalized values of f_{0:k} (0 <= k <= t - 1).
    '''
    
    ### YOUR CODE HERE ###
    
    return None


def backward_recursion(env: Environment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform the smoothing task for each time step.

    Calculate and return the values b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.

    :return: A numpy array with shape (t+1, N), (N is the number of states)
            the k'th row represents the normalized values of b_{k:t-1} (1 <= k <= t - 1),
            while the k=0 row is meaningless and we will NOT test it.
    '''

    ### YOUR CODE HERE ###
    return None


def fba(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array with shape (t,N) where t = len(observ) and N is the number of states.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env: The environment.
    :param actions: A list of agent's past actions.
    :param observ: A list of observations.
    :param probs_init: The agent's initial beliefs over states
    :return: A numpy array with shape (t, N)
        the k'th row represents the normalized smoothed probability distribution over all the states for time k.
    '''

    ### YOUR CODE HERE ###
    return None

