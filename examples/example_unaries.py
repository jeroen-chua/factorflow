"""Example of attaching multiple unary potentials to a single variable node.
"""

import sys
sys.path.append("..")

from graph import BpGraph
from nodesLib import VarNodes

var_nodes = VarNodes('input', {'num_states': 2})
var_nodes.create_nodes(1)

#add three unaries to variable nodes
var_nodes.add_unaries([0], [0.7, 0.3])
var_nodes.add_unaries([0], [0.4, 0.6])
var_nodes.add_unaries([0], [0.2, 0.8])

bpg = BpGraph()
bpg.add_nodes_to_schedule(var_nodes)

bpg.finalize()
bpg.do_message_passing()

print "variable node beliefs:"
print var_nodes.bel
