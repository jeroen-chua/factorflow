"""Example of using the PottsNodes class to denoise an image.
"""

import sys
sys.path.append("..")

import numpy as np
from skimage import color
import scipy as sp
import pylab
from bp_graph import BpGraph
from nodesLib import VarNodes
from nodesLib import PottsNodes
import utils

# model parameters
POTTS_ALPHA = 1e-3
NUM_STATES = 256

#load in an image
image = sp.misc.imread("noisy.png")
image = image.astype("float")/float(NUM_STATES-1)

#crop image
image = image[20:28, 20:28]
IM_SZ = image.shape[0:2]
SIG2 = 1.0/float(NUM_STATES)

#convert to graysale
image = color.rgb2gray(image)

#evaluate the image evidence. Gaussian with a variance of 1/256.
obs_dist = -np.power(np.expand_dims(image, 2)- \
                     np.arange(0, 256).reshape(1, 1, NUM_STATES)/float(NUM_STATES-1), \
                     2)/SIG2
obs_vals = np.exp(obs_dist-sp.misc.logsumexp(obs_dist, 2, keepdims=True))

#rasterize
obs_vals = np.reshape(obs_vals, [obs_vals.shape[0]*obs_vals.shape[1], obs_vals.shape[2]])

#create the factor graph object
bpg = BpGraph()

#create a container for variable nodes.
var_nodes = VarNodes('var', {'num_states': NUM_STATES})

node_ids = var_nodes.create_nodes(IM_SZ[0]*IM_SZ[1])
var_nodes.add_unaries(node_ids, obs_vals)
node_ids = np.reshape(node_ids, IM_SZ)

#create PottsNodes container running sum-product loopy belief propagation
potts_nodes = PottsNodes(nodes_params={'alpha': POTTS_ALPHA, 'bp_algo': 'max'})

#add a Potts factor between neighbouring variable nodes, and connect the
#variable nodes to the factor
for i in range(0, IM_SZ[0]):
    for j in range(0, IM_SZ[1]-1):
        factId = potts_nodes.create_nodes(1)
        bpg.add_edge(var_nodes, node_ids[i, j], potts_nodes, factId[0])
        bpg.add_edge(var_nodes, node_ids[i, j+1], potts_nodes, factId[0])

for i in range(0, IM_SZ[0]-1):
    for j in range(0, IM_SZ[1]):
        factId = potts_nodes.create_nodes(1)
        bpg.add_edge(var_nodes, node_ids[i, j], potts_nodes, factId[0])
        bpg.add_edge(var_nodes, node_ids[i+1, j], potts_nodes, factId[0])

# schedule nodes for message-passing in graph object
bpg.add_nodes_to_schedule(var_nodes)
bpg.add_nodes_to_schedule(potts_nodes)

#prepare graph object for inference
bpg.finalize()

#do inference
bpg.do_message_passing()

#get resulting beliefs
bel = var_nodes.get_beliefs()

#show original data and most likely state for each pixel
val = bel.argmax(axis=1)
val = np.reshape(val, IM_SZ)
val = val.astype('float')/(NUM_STATES-1)

pylab.subplot(1, 2, 1)
utils.imshow_gray(image)

pylab.subplot(1, 2, 2)
utils.imshow_gray(val, False)
