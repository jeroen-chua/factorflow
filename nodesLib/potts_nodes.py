"""Module for the PottsNodes class. Models Potts factor nodes. A Potts factor
is a function of two variables, X and Y, that have a discrete number of states.
Below is the definition of the factor:

F(X,Y) = \alpha  if X==Y,
       = 1       otherwise.

The model assumes \alpha > 0 and typically, \alpha < 1 (ie, variables tend to
prefer to be in the same state.).
"""

import numpy as np
from nodesLib.factor_nodes import FactorNodes

class PottsNodes(FactorNodes):
    """This class represents a collection of Potts factor nodes.
    Message-passing is performed using loopy belief propagation, either in the
    sum-product or max-product setting, as specified by the user. This class is
    a concrete implemention of the FactorNodes class. Potts factors have
    one kind of edge: "default".

    Public methods:
        finalize: Prepares contained variable nodes for message-passing.
    """

    EDGE_TYPES = frozenset({'default'})
    BP_ALGO_TYPES = frozenset({'max', 'sum'})

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
                    'alpha': value for F(X,Y) when X==Y. See module
                        documentation for description of Potts factor.
        """

        assert nodes_params is not None, 'nodes_params cannot be None for Potts factor'
        assert 'alpha' in nodes_params, 'nodes_params must contain key "alpha"'

        assert nodes_params['alpha'] > 0, 'alpha must be > 0.'

        assert 'bp_algo' in nodes_params.keys(), \
                          'No "bp_algo" parameters found for PottsNodes object.'

        assert nodes_params['bp_algo'] in self.BP_ALGO_TYPES, \
                   'Must specify valid bp_algo type.'

        super(PottsNodes, self).__init__(name, nodes_params)

    def _do_compute_messages(self):
        alpha = self.nodes_params['alpha']
        msgs = self.get_msgs_on_edge('default')
        res = np.zeros(msgs.shape)

        res[0, :, :] = msgs[1, :, :]
        res[1, :, :] = msgs[0, :, :]
        if self.nodes_params['bp_algo'] == 'sum':
            res = res*(1-alpha)+alpha
        else:
            #record where the max is
            inds = res.argmax(axis=1)

            #get maxmimum over all states
            state_max = alpha*np.max(res, axis=1)
            state_max = state_max[:, np.newaxis, :]

            #mask out the maximum value. for finding 2nd max
            mask = np.zeros_like(res)
            ind0 = np.asarray(range(0, res.shape[0]))
            ind0 = ind0[:, np.newaxis]

            ind2 = np.asarray(range(0, res.shape[2]))
            ind2 = ind2[np.newaxis, :]
            mask[ind0, inds, ind2] = 1

            res_masked = np.ma.masked_array(res, mask=mask)

            #find second-max
            state_max2 = alpha*np.max(res_masked, axis=1)

            #for all entries except the max-state, this is the correct value
            tmp_max = np.maximum(res, state_max)

            tmp_max[ind0, inds, ind2] = np.maximum(res[ind0, inds, ind2], state_max2)
            res = tmp_max

        res /= res.sum(axis=1, keepdims=True)
        return {'default': res}

    def finalize(self):
        """Prepares contained factor nodes for message passing and performs
        error-checking. Must be called once before message passing can be done
        on this object.
        """

        super(PottsNodes, self).finalize()
        assert self.nodes_params['bp_algo'] in self.BP_ALGO_TYPES, \
               'Internal error: Invalid BP algorithm specified?'

        for key in self.message_chunks:
            chunk = self.message_chunks[key]
            assert chunk.msgs_in.shape[0] == 2, str(chunk.msg_in.shape)
