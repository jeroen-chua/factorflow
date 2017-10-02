"""Example of using the NoisyOrNodes class.
"""

import numpy as np

from bp_graph import BpGraph
from nodesLib import VarNodes
from nodesLib import NoisyOrNodes

NUM_INPUTS = 5
NUM_STATES = 2

#create graph object
bpg = BpGraph()

node_ids = -np.ones(NUM_INPUTS, dtype="int")

#create NoisyOrNodes container running sum-product loopy belief propagation and
#create a NoisyOr factor node.
noisy_or_nodes = NoisyOrNodes(nodes_params={'leak_prob': 0.01, \
                                            'prob_success': 0.99, \
                                            'bp_algo': 'sum'})
fact_id = noisy_or_nodes.create_nodes(1)[0]

#create containers for inputs and outputs variables
var_inputs = VarNodes('inputs', {'num_states': NUM_STATES})
var_output = VarNodes('output', {'num_states': NUM_STATES})

#create input nodes and connect output them to the LeakyOr factor node
for i in range(0, NUM_INPUTS):
    tmp_id = var_inputs.create_nodes(1)[0]
    #if i == 0:
    #    var_inputs.add_unaries([tmp_id],[1,0])

    bpg.add_edge(var_inputs, tmp_id, noisy_or_nodes, fact_id, 'input')

#create an output node and connect it to the LeakyOr factor node
tmp_id = var_output.create_nodes(1)[0]
bpg.add_edge(var_output, tmp_id, noisy_or_nodes, fact_id, 'output')

# schedule nodes for message-passing in graph object
bpg.add_nodes_to_schedule(var_inputs)
bpg.add_nodes_to_schedule(var_output)
bpg.add_nodes_to_schedule(noisy_or_nodes)

#prepare graph object for inference
bpg.finalize()

#do inference
bpg.do_message_passing()

#get resulting beliefs
bel_in = var_inputs.get_beliefs()
bel_out = var_output.get_beliefs()
