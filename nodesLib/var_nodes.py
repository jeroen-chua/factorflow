"""Module for the VarNodes class. See documentation for VarNodes class."""

import numpy as np
import scipy as sp
from nodesLib import MessageChunk
from nodesLib.nodes import Nodes

class VarNodes(Nodes):
    """This class represents a collection of variable nodes in a factor graph
    and provides methods to perform operations on the messages of these nodes.
    Outgoing messages are computed as in the sum-product belief propagation
    algorithm. This class is a concrete implemention of the Nodes class.

    Public methods:
        get_msg_chunk: Returns the Chunk of variable nodes this object
        represents

        finalize: Prepares contained variable nodes for message-passing.

        add_unaries: Adds unary potentials to the specified variable nodes.

        get_beliefs: Returns the beliefs of the contained variable nodes.

        condition_on: Condition on the state of a variable node.
    """

    __DEFAULT_PARAMS = {'num_states': 2, \
                        'msgs_init_strat': MessageChunk.MSGS_INIT_ENUM.random}

    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this VarNodes object. Defaults
                 to the emptry string.

            nodes_params (dict, optional): parameters for this VarNodes
                instance. If none are provided, default values are used. See
                VarNodes.__DEFAULT_PARAMS for default values.
        """

        #don't set default to empty dictionary. Create a new dictionary each time.
        if nodes_params is None:
            nodes_params = {}

        for field in VarNodes.__DEFAULT_PARAMS:
            if field not in nodes_params:
                nodes_params[field] = VarNodes.__DEFAULT_PARAMS[field]

        super(VarNodes, self).__init__(name, nodes_params)

        self.bel = []
        self.num_states = nodes_params['num_states']
        self.message_chunks['vars'] = MessageChunk(name+'_vars', self.num_states)
        self.message_chunks['vars'].msgs_init_strat = nodes_params['msgs_init_strat']

    def condition_on(self, node_ids, state):
        """Condition on the state of given variable nodes.

        Args:
            node_ids (list): A list of the ids of the variable nodes (as
            returned by create_nodes) that is to be conditioned on.

            state: state (numbered from 0) to condition on. All given
                   variable nodes will be condition to be in the given state.
        """

        assert state < self.num_states

        for node_id in node_ids:
            tmp = [0.0]*self.num_states
            tmp[state] = 1.0
            self.add_unaries([node_id], tmp)

    def get_msg_chunk(self):
        """ Returns the MessageChunk of variables nodes this object represents."""

        return self.message_chunks['vars']

    def finalize(self):
        """Prepares contained variable nodes for message-passing.

        Preparation involves allocating space for and initializing messages.
        """

        #do we have a unary attached to all nodes? then sort and delete all indices.
        #this lets us avoid fancy indexing since we know the observations are in a
        #contiguous chunk
        if 'unary_idx' in self.nodes_params:
            if self.nodes_params['unary_idx'].size == self.message_chunks['vars'].num_nodes:
                idx_resh = np.argsort(self.nodes_params['unary_idx'], axis=0)
                self.nodes_params['log_unary'] = self.nodes_params['log_unary'][:, :, idx_resh]
                del self.nodes_params['unary_idx']

        super(VarNodes, self).finalize()

    def add_unaries(self, node_ids, unary_vals):
        """Adds unary potentials to the specified variable nodes.

        Args:
            node_ids (list): A list of the ids of the variable nodes (as
            returned by create_nodes) to which unary potentials will be
            attached.

            unary_vals (ndarray or list): Generally, an ndarray of doubles of
            size [len(node_ids), self.num_states] indicating the values of the
            unary potentials. the ith row defines the unary potential for the
            variable node with id node_ids[i]. In the special case of only
            one node being specified in node_ids, then unary_vals can be a list
            of values with len(unary_vals)==self.num_states.
        """

        if isinstance(unary_vals, (list, tuple)):
            assert len(node_ids) == 1, 'Can only specify one variable node ' + \
            'to attach a unary potential to if unary_vals is a list of values.'

            unary_vals = np.asarray(unary_vals)
            unary_vals = np.reshape(unary_vals, [1, unary_vals.size])

        assert unary_vals.ndim == 2, 'unary_vals must be a 2D ndarray'

        assert unary_vals.shape[0] == len(node_ids), \
                                      'Must specify unary potential for ' + \
                                      'each specified variable node.'

        assert unary_vals.shape[1] == self.message_chunks['vars'].num_states, \
                                      'Unary potentials must specify ' + \
                                      'same number of states as node.'

        unary_to_reshape = [unary_vals.shape[0], unary_vals.shape[1], 1]
        unary_vals = np.reshape(unary_vals, unary_to_reshape)
        unary_vals = unary_vals.swapaxes(0, 2)
        node_ids = np.asarray(node_ids)
        node_ids = np.reshape(node_ids, [node_ids.size])

        unary_vals /= np.sum(unary_vals, axis=1, keepdims=True)

        #clip for numerical issues
        unary_vals = np.clip(unary_vals, 1e-12, 1-1e-12)
        unary_vals /= np.sum(unary_vals, axis=1, keepdims=True)

        if 'log_unary' in self.nodes_params:
            (node_ids, unary_vals) = self.__merge_duplicate_unaries(node_ids, unary_vals)

            if node_ids.size > 0:
                self.nodes_params['log_unary'] = np.concatenate((self.nodes_params['log_unary'], \
                                                                np.log(unary_vals)), \
                                                                axis=2)
                self.nodes_params['unary_idx'] = np.concatenate((self.nodes_params['unary_idx'], \
                                                               node_ids), \
                                                               axis=0)
        else:
            self.nodes_params['log_unary'] = np.log(unary_vals)
            self.nodes_params['unary_idx'] = np.copy(node_ids)

    def __merge_duplicate_unaries(self, node_ids, unary_vals):
        # merges duplicate unaries and returns the variable nodes that are
        # not duplicate entries (ie, that do not yet have unary potentials
        # attached to them.).

        dups = []
        dups_val = None
        for i in range(0, len(node_ids)):
            node_id = node_ids[i]
            if node_id in self.nodes_params['unary_idx']:
                dups = np.append(dups, node_id)

                if dups_val is None:
                    dups_val = unary_vals[[i], :, :]
                else:
                    dups_val = np.concatenate((dups_val, unary_vals[[i], :, :]), axis=0)

        if dups.size > 0:
            log_dups_val = np.log(dups_val)

            for i in dups:
                idx = np.where(self.nodes_params['unary_idx'] == i)[0][0]
                self.nodes_params['log_unary'][idx, :] += log_dups_val[int(i), :]

        node_ids = np.delete(node_ids, dups)
        unary_vals = np.delete(unary_vals, (dups), axis=0)

        return (node_ids, unary_vals)

    def __include_unary(self, log_arr):
        """Adds on the effect of the (log of) unary potentials.

            Args:
                log_arr (double): an ndarray of doubles the same size as
                    self.log_unary

            Returns:
                arr modified to include information about the unary potentials
        """

        if 'log_unary' in self.nodes_params:
            if 'unary_idx' in self.nodes_params:
                unary_idx = self.nodes_params['unary_idx']
                log_arr[:, :, unary_idx] += self.nodes_params['log_unary']
            else:
                log_arr += self.nodes_params['log_unary']
        return log_arr

    def get_beliefs(self):
        """Returns the beliefs of the contained variable nodes.

            Returns:
                Belief of the variable nodes as an ndarray
        """

        self.__update_beliefs()
        return self.bel

    def __update_beliefs(self):
        """Updates the beliefs of the contained variable nodes."""

        log_bel = np.sum(np.log(self.message_chunks['vars'].msgs_in), \
                         axis=0, \
                         keepdims=True)
        log_bel = self.__include_unary(log_bel)

        denom = sp.misc.logsumexp(log_bel, axis=1, keepdims=True)
        self.bel = np.exp(log_bel - denom)

    def _do_compute_messages(self):
        """Helper function to compute messages from variable nodes.

        Returns:
            dict with key 'vars', containing the computed messages
        """

        msg_in = self.message_chunks['vars'].msgs_in

        ###OPTIMIZE FOR DEGREE 2
        if msg_in.shape[0] == 2:
            f_msg = np.zeros(msg_in.shape)
            f_msg[0, :, :] = msg_in[1, :, :]
            f_msg[1, :, :] = msg_in[0, :, :]

        else:
            log_mess = np.log(msg_in)
            all_log_sum = np.sum(log_mess, axis=0, keepdims=True)
            self.__include_unary(all_log_sum)
#
            f_msg = all_log_sum-log_mess
            denom = sp.misc.logsumexp(f_msg, axis=1, keepdims=True)
            f_msg = np.exp(f_msg - denom)

        return {'vars': f_msg}
