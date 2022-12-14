
# Cream ( CoRy Ego Ai Model )
# Version: 0.1.0
# Developer: @mangto (github)
# Contact: mangto0701@gmail.com

import cream.engine.cream_beta as cream
import cream.engine.one_neuron_snn as onsnn
import cream.engine.SNN as snn
from cream.engine.integrated import network as network
import cream.engine.word2vec as word2vec
import cream.engine.recurrent_neural_network as rnn
from cream.engine.reinforcement import network as reinforce

import cream.tool.Csys as csys
import cream.tool.datasets as datasets
import cream.tool.progress_bar as progress_bar

import cream.Functions as functions
import cream.Functions.cnn.kernel as kernel
from cream.Functions.cnn.convolution import *
import cream.Functions.cnn.pooling as pool
from cream.Functions.cnn.cv2 import *
from cream.Functions.cnn.padding import *

import cream.visualizer

print(f"CREAM Beta Version 0.1.0 by @mangto\nIf you find bugs or something to change, please contact 'mangto0701@gmail.com'")