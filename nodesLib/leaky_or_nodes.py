"""Module for the LeakyOrNodes class. Models leaky-or factor nodes. A leaky-or
distribution has many "input" variable nodes, and a one "output" variable node.
All variable nodes are binary-valued. Let Y be the set of input variables
nodes, and let c(Y) indicate how many input variable nodes take value 1.
Let z represent the output variable node. A leaky-or distribution is defined
by:

p(z=1|c(Y)=0) = 1
p(z=0|c(Y)=0) = \epsilon

where \epsilon is the "leak" parameter.
"""

import numpy as np
from nodesLib.factor_nodes import FactorNodes
from nodesLib import MessageChunk

class LeakyOrNodes(FactorNodes):
    """This class represents a collection of LeakyOr factor nodes.
    Message-passing is performed using loopy belief propagation (sum-product).
    This class is a concrete implemention of the FactorNodes class.

    Public methods:
        finalize: Prepares contained variable nodes for message-passing.
    """

    EDGE_TYPES = frozenset({'input', 'output'})

    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this LeakOrNodes object.
                 Defaults to the emptry string.

            nodes_params (dict): parameters for this CatNodes. Mandatory
                keys are:
                    'leak_prob': specifies the quantity p(z=1|c(Y)=0). Must be
                    in range [0,1].
        """

        assert nodes_params is not None, \
               'Must specify success parameters for making LeakyOrNodes object.'

        assert 'leak_prob' in nodes_params.keys(), \
                          'No "leak_prob" parameter found for LeakyOrNodes object.'
        assert nodes_params['leak_prob'] >= 0 and \
               nodes_params['leak_prob'] <= 1, \
               'Invalid value for leak_prob.'

        super(LeakyOrNodes, self).__init__(name, nodes_params)

        self.message_chunks['input'].msg_init_strat = MessageChunk.MSGS_INIT_ENUM.random

        #setup padded message values so non-existent input variable nodes have
        # no effect.
        self.message_chunks['input'].pad_msg_val = np.asarray([1.0, 0.0])
        self.message_chunks['output'].pad_msg_val = np.asarray([0.5, 0.5])

        self.message_chunks['input'].msg_low = [1-nodes_params['leak_prob'], \
                                                nodes_params['leak_prob']]
        self.message_chunks['input'].msg_range = [0.0, 0.0]

    def _do_compute_messages(self):

        res = {}

        leak_prob = self.nodes_params['leak_prob']

        msg_from_input = self.get_msgs_on_edge('input')
        msg_from_output = self.get_msgs_on_edge('output')

        #compute message to output
        prod_msg_0 = np.prod(msg_from_input[:, [0], :], axis=0, keepdims=True)

        msg = np.zeros(msg_from_output.shape)
        msg[:, [0], :] = (1-leak_prob)*prod_msg_0
        msg[:, [1], :] = 1-msg[:, [0], :]

        res['output'] = msg
        #compute message to output

        #compute message to input
        msg = np.zeros(msg_from_input.shape)
        msg[:, [1], :] = msg_from_output[:, [1], :]

        r_j = prod_msg_0 / msg_from_input[:, [0], :]

        diff = (msg_from_output[:, [0], :] - msg_from_output[:, [1], :])
        msg[:, [0], :] = msg_from_output[:, [1], :] + (1-leak_prob)*r_j*diff


        msg = msg / np.sum(msg, axis=1, keepdims=True)

        res['input'] = msg

        return res
        #compute message to input

    def finalize(self):
        """Prepares contained leaky-or factor nodes for message passing and
        performs error-checking. Must be called once before message passing can
        be done on this object.
        """

        super(LeakyOrNodes, self).finalize()
