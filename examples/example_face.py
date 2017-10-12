"""An example of using sum-product loopy belief propagation to reason about
faces and their parts in an image."""

import sys
sys.path.append("..")

import numpy as np
import pylab

from graph import BpGraph
from nodesLib import VarNodes
from nodesLib import NoisyOrNodes
from nodesLib import CatNodes
import matplotlib.pyplot as plt

def imshow_gray(image, rescale=False):
    """ Displays an image in gray-scale.

    Args:
        image (ndarray): a 2D array representing the image

        rescale (boolean, optional): rescale the image so entries are in the
            range [0,1].
    """
    if not rescale:
        assert np.max(image) <= 1, 'max value must be <= 1'
        assert np.min(image) >= 0, 'max value must be >= 0'
        plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, interpolation='None')
    else:
        plt.imshow(image, cmap=plt.get_cmap('gray'), interpolation='None')

    plt.draw()
    plt.pause(0.001)

#image size
IM_SZ = [35, 35]

#condition on the presence of a symbol(s) at a particular place(s) in the image.
#uncomment one of the 3 lines below to see different scenarios.
COND_POINTS = [['face', 18, 18]]
#COND_POINTS = [['eye', 18, 18]]
#COND_POINTS = [['face', 18, 18], ['eye', 12, 16]]

#symbols in our model. Note: we do not distinguish between left and right eyes.
SYMBOLS = ('face', 'eye', 'nose', 'mouth')

#parts of the face. the model encodes the production rule
#  ace-->eye,eye,nose,mouth
FACE_PARTS = ('eye', 'eye', 'nose', 'mouth')

#for each of the FACE_PARTS, specify where they are located relative to the
#face and how large the region of uncertainties are.
OFFSETS = [[-7, -3], [-7, 3], [0, 0], [7, 0]]
REGION_SIZES = [[3, 3], [3, 3], [3, 3], [7, 3]]

#used to keep track of the nodes (variable nodes, etc.) associated with
#a particular symbol (eg, face).
sym_super_nodes = {}

bpg = BpGraph()

#create variable node containers and noisy-or factor containers. We will add
#variable and noisy-or factor nodes to these containerslater.
for i in range(0, len(SYMBOLS)):
    sym = SYMBOLS[i]
    tmp_dict = {}
    tmp_dict['vars'] = VarNodes(sym+'_vars', {'num_states': 2})
    tmp_dict['noisy'] = NoisyOrNodes(name=sym + '_noisy',
                                     nodes_params={'leak_prob': 0.01, \
                                                   'prob_success': 0.99, \
                                                   'bp_algo': 'sum'})

    tmp_dict['var_ids'] = np.zeros(IM_SZ, dtype='int')
    tmp_dict['fact_ids'] = np.zeros(IM_SZ, dtype='int')

    sym_super_nodes[sym] = tmp_dict

#create variable nodes for presene/absence of each object and its connecting
#noisy-or factor. Cconnect these entities together.
for sym in SYMBOLS:
    chunk_dict = sym_super_nodes[sym]

    for i in range(0, IM_SZ[0]):
        for j in range(0, IM_SZ[1]):

            chunk_dict['var_ids'][i, j] = chunk_dict['vars'].create_nodes(1)[0]

            chunk_dict['fact_ids'][i, j] = chunk_dict['noisy'].create_nodes(1)[0]
            bpg.add_edge(chunk_dict['vars'], chunk_dict['var_ids'][i, j], \
                         chunk_dict['noisy'], chunk_dict['fact_ids'][i, j], \
                         'output')

face_super_nodes = sym_super_nodes['face']

#express for a face each location in the image, where its parts (eye, eye,
#nose, mouth) could be.
for ch_ind in range(0, len(FACE_PARTS)):
    face_part = FACE_PARTS[ch_ind]
    offset = OFFSETS[ch_ind]

    #parameters for a CatNode factor (representing a categorical distribution)
    # o represent the spatial relationship between a face and this part
    #
    num_choices = REGION_SIZES[ch_ind][0]*REGION_SIZES[ch_ind][1]+1
    probs_use = np.ones((2, num_choices))/(REGION_SIZES[ch_ind][0]*REGION_SIZES[ch_ind][1])
    probs_use[0, :] = 0.0
    probs_use[0, -1] = 1.0
    probs_use[1, -1] = 0.0

    #create CatNodes container for storing categorical factors. Create
    #variable nodes representing the possible outcomes of the categorical
    #factors. Encoding uses a 1-hot encoding for the output choice of a
    #categorical factor.
    tmp_cat_nodes = CatNodes(name=face_part + '_cat' + str(ch_ind),
                             nodes_params={'probs': probs_use, \
                                           'bp_algo': 'sum'})

    tmp_cat_vars = VarNodes(face_part + '_vars_cat' + str(ch_ind), {'num_states': 2})

    ch_chunk = sym_super_nodes[face_part]
    for i in range(0, IM_SZ[0]):
        for j in range(0, IM_SZ[1]):

            #create categorical node
            cat_id = tmp_cat_nodes.create_nodes(1)[0]
            bpg.add_edge(face_super_nodes['vars'], \
                         face_super_nodes['var_ids'][i, j], \
                         tmp_cat_nodes, \
                         cat_id, \
                         'input')

            #create categorical variable nodes
            cat_var_ids = tmp_cat_vars.create_nodes(num_choices)
            for c_id in cat_var_ids:
                bpg.add_edge(tmp_cat_vars, \
                             c_id, \
                             tmp_cat_nodes, \
                             cat_id, \
                             'output')

            #hook up categorical variables to noisy-or factors
            ct = 0
            for ii in range(0, REGION_SIZES[ch_ind][0]):
                ii_use = ii+i+offset[0]

                if ii_use < 0 or ii_use >= IM_SZ[0]:
                    continue

                for jj in range(0, REGION_SIZES[ch_ind][1]):
                    jj_use = jj+j+offset[1]

                    if jj_use < 0 or jj_use >= IM_SZ[1]:
                        continue

                    bpg.add_edge(tmp_cat_vars, \
                                 cat_var_ids[ct], \
                                 ch_chunk['noisy'], \
                                 ch_chunk['fact_ids'][ii_use, jj_use], \
                                'input')
                    ct += 1

    bpg.add_nodes_to_schedule(tmp_cat_nodes)
    bpg.add_nodes_to_schedule(tmp_cat_vars)

#condition on presence/absence of certain symbols in the scene at certain
#locations
for cond_pt in COND_POINTS:
    [sym, i, j] = cond_pt
    sym_super = sym_super_nodes[sym]
    sym_super['vars'].condition_on([sym_super['var_ids'][i, j]], 1)

# schedule nodes for message-passing in graph object
for i in SYMBOLS:
    bpg.add_nodes_to_schedule(sym_super_nodes[i]['vars'])
    bpg.add_nodes_to_schedule(sym_super_nodes[i]['noisy'])

#prepare graph object for inference
bpg.finalize()

#do inference
bpg.do_message_passing()

#get resulting beliefs
for i in SYMBOLS:
    sym_super_nodes[i]['bel'] = sym_super_nodes[i]['vars'].get_beliefs()
    sym_super_nodes[i]['bel'] = np.reshape(sym_super_nodes[i]['bel'][:, 1], IM_SZ)

#visualize resulting beliefs on a log-scale, and normalize to be in range [0,1]
for i in range(0, len(SYMBOLS)):
    pylab.subplot(1, len(SYMBOLS), i+1)
    imshow_gray(np.log(sym_super_nodes[SYMBOLS[i]]['bel']), True)
