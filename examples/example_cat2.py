"""Example of using the CatNodes class and condition on the state of a
variable node.
"""

import sys
sys.path.append("..")

import numpy as np
from graph import BpGraph
from nodesLib import VarNodes
from nodesLib import CatNodes

NUM_OUTPUTS = 5
NUM_INPUT_STATES = 3
NUM_OUPUT_STATES = 2

#create factor graph object
bpg = BpGraph()

node_ids = -np.ones(NUM_OUTPUTS, dtype="int")

#success parameters for categorical. 3 states for input, 5 possible outputs.
PROBS = [[0.3, 0.5, 0.1, 0.05, 0.05], \
         [0.0, 0.0, 0.0, 0.00, 1.00], \
         [0.4, 0.4, 0.1, 0.05, 0.05]]

#create CatNodes container running sum-product loopy belief propagation and
#create a categorical factor node.
cat_nodes = CatNodes(nodes_params={'probs': np.asarray(PROBS), 'bp_algo': 'sum'})
fact_id = cat_nodes.create_nodes(1)[0]

#create containers for inputs and outputs variables
var_input = VarNodes('inputs', {'num_states': NUM_INPUT_STATES})
var_outputs = VarNodes('outputs', {'num_states': NUM_OUPUT_STATES})

#create output nodes and connect output them to the categorical factor node
for i in range(0, NUM_OUTPUTS):
    tmp_id = var_outputs.create_nodes(1)[0]
    bpg.add_edge(var_outputs, tmp_id, cat_nodes, fact_id, 'output')

#create an input node and connect it to the categorical factor node
tmp_id = var_input.create_nodes(1)
bpg.add_edge(var_input, tmp_id[0], cat_nodes, fact_id, 'input')
#var_input.add_unaries(tmp_id, [0, 0, 1])

#condition on the state of an output node
var_outputs.condition_on([4], 1)

# schedule nodes for message-passing in graph object
bpg.add_nodes_to_schedule(var_input)
bpg.add_nodes_to_schedule(var_outputs)
bpg.add_nodes_to_schedule(cat_nodes)

#prepare graph object for inference
bpg.finalize()

#do inference
bpg.do_message_passing()

#get resulting beliefs
bel_in = var_input.get_beliefs()
bel_out = var_outputs.get_beliefs()

print "input beliefs:"
print bel_in

print "output beliefs:"
print bel_out