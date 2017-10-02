"""Module for the NoisyOrNodes class. Models noisy-or factor nodes. A noisy-or
distribution has many "input" variable nodes, and a one "output" variable node.
All variable nodes are binary-valued. Let Y be the set of input variables
nodes, and let c(Y) indicate how many input variable nodes take value 1.
Let z represent the output variable node. A y-or distribution is defined
bynoisy

p(z=0|c(Y)=0) = (1-\epsilon)(1-\rho)^{c(Y)}
p(z=1|c(Y)=0) = 1 - p(z=0|c(Y)=0)

where \epsilon is the "leak" parameter and \rho is the probability of an input
successfully turning the output on. \epsilon represents the probability the
output "turns itself" on, and \rho is the probability than an input turns the
output on. Note that the output is on if any of the inputs turns the output
on (hence, this is a noisy-or).
"""

import numpy as np
from nodesLib.factor_nodes import FactorNodes

class NoisyOrNodes(FactorNodes):
    """This class represents a collection of noisy-or factor nodes.
    Message-passing is performed using loopy belief propagation, either in the
    sum-product or max-product setting, as specified by the user. This class is
    a concrete implemention of the FactorNodes class. NoisyOr factors have
    two kinds of edges: "input"and "output".

    Public methods:
        finalize: Prepares contained variable nodes for message-passing.
    """

    EDGE_TYPES = frozenset({'input', 'output'})
    BP_ALGO_TYPES = frozenset({'max', 'sum'})

    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this NoisyOrNodess object.
            Defaults to the emptry string.

            nodes_params (dict): parameters for this NoisyOrNodess object.
                Mandatory keys are:
                    'bp_algo': takes valye either 'max' or 'sum' corresponding
                        to the type of loopy belief propagation that is to be
                        used.

                    'leak_prob': specifies the leak probability (\epsilon in
                        the module definition). Must be in range [0,1].

                    'prob_success': the probability than an input turns the
                        output on (\rho in the module definition). Must be in
                        the range [0,1].
        """

        assert nodes_params is not None, \
               'Must specify success parameters for making NoisyOrNodes object.'

        assert 'leak_prob' in nodes_params.keys(), \
                          'No "probs" parameter found for NoisyOrNodes object.'
        assert nodes_params['leak_prob'] >= 0 and \
               nodes_params['leak_prob'] <= 1, \
               'Invalid value for leak_prob.'

        assert 'prob_success' in nodes_params.keys(), \
                          'No "prob_success" parameter found for NoisyOrNodes object.'
        assert nodes_params['prob_success'] >= 0 and \
               nodes_params['prob_success'] <= 1, \
               'Invalid value for prob_success.'


        assert 'bp_algo' in nodes_params.keys(), \
                          'No "bp_algo" parameters found for NoisyOrNodes object.'

        assert nodes_params['bp_algo'] in self.BP_ALGO_TYPES, \
                   'Must specify valid bp_algo type'

        if nodes_params['bp_algo'] == 'max':
            self.max_or_sum = np.max
        else:
            self.max_or_sum = np.sum

        super(NoisyOrNodes, self).__init__(name, nodes_params)

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
        exp_neg_beta = (1-self.nodes_params['prob_success'])

        msg_from_input = self.get_msgs_on_edge('input')
        msg_from_output = self.get_msgs_on_edge('output')

        #compute message to output
        tmp_weighted = msg_from_input[:, [0], :] + \
                       exp_neg_beta*msg_from_input[:, [1], :]

        tmp_prod_weighted = np.prod(tmp_weighted, axis=0, keepdims=True)

        msg = np.zeros(msg_from_output.shape)

        msg[:, [0], :] = (1-leak_prob)*tmp_prod_weighted
        msg[:, [1], :] = 1-msg[:, [0], :]

        res['output'] = msg
        #compute message to output

        #compute message to input
        r_j = tmp_prod_weighted / tmp_weighted

        tmp_out = (1-leak_prob)*r_j*(msg_from_output[:, [0], :] - msg_from_output[:, [1], :])

        msg = np.zeros(msg_from_input.shape)

        msg[:, [0], :] = msg_from_output[:, [1], :] + tmp_out
        msg[:, [1], :] = msg_from_output[:, [1], :] + exp_neg_beta*tmp_out

        msg = msg / np.sum(msg, axis=1, keepdims=True)

        res['input'] = msg

        return res
        #compute message to input

    def finalize(self):
        """Prepares contained factor nodes for message passing and performs
        error-checking. Must be called once before message passing can be done
        on this object.
        """

        super(NoisyOrNodes, self).finalize()
