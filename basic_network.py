# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 17:59:38 2025

@author: Daniel De la Cueva
"""

import numpy as np

class Network(object):
    """
        Python class representing a neural network. Takes a list representing the amount of neurons in each layer
    """
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weighs = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]