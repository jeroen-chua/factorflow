"""Module for the CatNodes class. Models categorical factor nodes.
"""

import numpy as np
from nodesLib.factor_nodes import FactorNodes

class CatNodes(FactorNodes):
    """This class represents a collection of categorical factor nodes.
    Message-passing is performed using loopy belief propagation, either in the
    sum-product or max-product setting, as specified by the user. This class is
    a concrete implemention of the FactorNodes class. Categorical factors have
    two kinds of edges: "input"and "output".

    Public methods:
        finalize: Prepares contained variable nodes for message-passing.
    """

    EDGE_TYPES = frozenset({'input', 'output'})
    BP_ALGO_TYPES = frozenset({'max', 'sum'})

    #nodes_params[probs]: a list of lists of success parameters. The list
    #probs[i] is the list of parameter values for the setting when the control
    #variable is i. probs assigned to output edges in order of adding
    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this CatNodes object. Defaults
                 to the emptry string.

            nodes_params (dict): parameters for this CatNodes object. Mandatory
                keys are:
                    'bp_algo': takes valye either 'max' or 'sum' corresponding
                        to the type of loopy belief propagation that is to be
                        used.
                    'probs': an nxm array of success parameters. probs[n][m]
                        is the probability of choosing output m if the input
                        variable takes value n. Success parameters are shared
                        by all categorica factors in this object instance.
                        Success parameters are assigned to output nodes in the
                        order they are connected to a categorical factor node.
        """

        assert nodes_params is not None, \
               'Must specify success parameters for making CatNodes object.'

        assert 'probs' in nodes_params.keys(), \
                          'No "probs" parameter found for CatNodes object.'

        assert 'bp_algo' in nodes_params.keys(), \
                          'No "bp_algo" parameters found for CatNodes object.'

        assert nodes_params['bp_algo'] in self.BP_ALGO_TYPES, \
                   'Must specify valid bp_algo type.'

        if nodes_params['bp_algo'] == 'max':
            self.max_or_sum = np.max
        else:
            self.max_or_sum = np.sum

        probs = nodes_params['probs']
        probs = np.asarray(probs)
        probs = probs.transpose()
        if probs.ndim < 2:
            probs = probs[:, np.newaxis]

        nodes_params['probs'] = np.reshape(probs, [probs.shape[0], probs.shape[1], 1])

        super(CatNodes, self).__init__(name, nodes_params)

    def _do_compute_messages(self):
        probs = self.nodes_params['probs']
        res = {}

        #compute message to input
        msg_from_outputs = self.get_msgs_on_edge('output')

        output_ratio = (msg_from_outputs[:, [1], :] / msg_from_outputs[:, [0], :])

        weighted_out = probs*output_ratio
        msg = self.max_or_sum(weighted_out, axis=0, keepdims=True)

        res['input'] = msg
        #compute message to input

        #compute message to outputs
        msg_from_input = self.get_msgs_on_edge('input')

        weighted_in = probs*msg_from_input
        msg = np.zeros(msg_from_outputs.shape)

        msg[:, [1], :] = self.max_or_sum(weighted_in, axis=1, keepdims=True)


        msg[:, [0], :] = output_ratio*msg[:, [1], :]
        msg[:, [0], :] = self.max_or_sum(msg[:, [0], :], axis=0, keepdims=True) - msg[:, [0], :]

        msg /= np.sum(msg, axis=1, keepdims=True)

        res['output'] = msg
        #compute message to outputs

        return res

    def finalize(self):
        """Prepares contained factor nodes for message passing and performs
        error-checking. Must be called once before message passing can be done
        on this object.
        """

        super(CatNodes, self).finalize()
        output_deg = self.message_chunks['output'].msgs_in.shape[0]
        probs_deg = self.nodes_params['probs'].shape[0]
        assert output_deg == probs_deg, \
               "Number of outputs of categorical does not match dimension of " + \
                "probs argument: %d vs %d" %(output_deg, probs_deg)
