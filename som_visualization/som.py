#### Last updated: 2016/11/30
#### This code is slightly modified from https://github.com/JustGlowing/minisom/blob/master/minisom.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from warnings import warn

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(np.dot(x, x.T))

class som(object):
    def __init__(self, x, y, input_len, sigma=None, learning_rate=1.0, decay_function=None, random_seed=None):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration)
            decay_function, function that reduces learning_rate and sigma at each iteration
                            default function: lambda x,current_iteration,max_iter: x/(1+current_iteration/max_iter)
            random_seed, random seed to use.
        """
        if sigma:
            if sigma >= x/2.0 or sigma >= y/2.0:
                warn('Warning: sigma is too high for the dimension of the map.')
            self.sigma = sigma
        else:
            self.sigma = min(x,y)/2.0
            
        if random_seed:
            self.random_generator = np.random.RandomState(random_seed)
        else:
            self.random_generator = np.random.RandomState(0)
            
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, T: x*np.exp(-(t*1.0)/(T*1.0))
        
        self.learning_rate = learning_rate
        
        self.weights = self.random_generator.rand(x,y,input_len)*2-1 # random initialization in range [-1,1]
        for i in range(x):
            for j in range(y):
                self.weights[i,j] = self.weights[i,j] / fast_norm(self.weights[i,j]) # normalization
        self.activation_map = np.zeros((x,y))
        self.neigx = np.arange(x)
        self.neigy = np.arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = np.subtract(x, self.weights)
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  # || x - w ||
            it.iternext()

    def gaussian(self, c, sigma):
        """ Returns a Gaussian centered in c """
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self.neigx-c[0], 2)/d)
        ay = np.exp(-np.power(self.neigy-c[1], 2)/d)
        #print(outer(ax, ay).shape)
        return np.outer(ax, ay)  # the external product gives a matrix

    def winner(self, x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
            T - a time constant
        """
        lr = self._decay_function(self.learning_rate, t, self.T)
        sig = self._decay_function(self.sigma, t, self.T) # sigma and learning rate decrease with the same rule
        
        g = self.neighborhood(win, sig)*lr
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])
            it.iternext()

    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[self.random_generator.randint(len(data))]
            it.iternext()

    def train_batch(self, data, num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(num_iteration)
        iteration = 0
        while iteration < num_iteration:
            #idx = iteration % (len(data)-1)
            idx = iteration % len(data)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """ Initializes the parameter time constant T needed to adjust the learning rate """
        #self.T = num_iteration/2  # keeps the learning rate nearly constant for the last half of the iterations
        #self.T = num_iteration/np.log(self.sigma)
        self.T = num_iteration  # keeps the learning rate nearly constant for the last half of the iterations

# ==================================================================================
