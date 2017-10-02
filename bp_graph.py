"""Module for the BpGraph class. See documentation for BpGraph class."""

import time
import math
import numpy as np
from nodesLib import VarNodes
from nodesLib import MessageChunk
from graph_edge_info import GraphEdgeInfo

class BpGraph(object):
    """This class represents a factor graph as a collection of nodes. It
    can be used to perform message-passing algorithms (e.g., loopy belief
    propagation in the sum-product or max-product sense) on the underlying
    factor graph.

    Public methods:
        add_nodes_to_schedule: adds a Nodes instance to the message-passing
            schedule.

        add_edge: adds an edge between a variable node and a factor node.

        get_scheduled_nodes: get the Node instances in this factor graph
            scheduled for message-passing.

        finalize: prepares the factor graph for message-passing.

        do_message_passing: performs the underlying message-passing algorithm
            (e.g., sum-product) on the factor graph.
    """

    __DEFAULT_PARAMS = {'iters': 1000, 'damp': 0.8, 'streak_lim': 10, 'tol': 1e-4}

    def __init__(self, bp_params=None):
        """Initializer.

        Args:
            bp_params (dict, optional): specifies parameters for
                message-passing. Any unspecified parameter takes its default
                value. See __DEFAULT_PARAMS for a listing of parameters and their
                default values.
        """

        self.graph_edge_info = GraphEdgeInfo()
        self.streak_count = 0

        if bp_params is None:
            self.bp_params = {}
        else:
            self.bp_params = bp_params

        #fill in rest of bp values
        for field in BpGraph.__DEFAULT_PARAMS:
            if field not in self.bp_params:
                self.bp_params[field] = BpGraph.__DEFAULT_PARAMS[field]

        self.prev_bel = []
        self.bel = []
        self.nodes = []
        self.__is_finalized = False

    def add_nodes_to_schedule(self, nodes):
        """Adds a Nodes instance to the message-passing schedule.

        Args:
            nodes (:obj: Nodes): a Nodes instance
        """

        if nodes not in self.nodes:
            self.nodes.append(nodes)

    def add_edge(self, c_nodes, c_id, o_nodes, o_id, edge_type=None):
        """Adds an edge between a variable node and a factor node.

        Args:
            c_nodes (:obj: VarNodes): a set of variable nodes

            c_id: the id of the variable node from c_nodes that will be at one
                end of the edge

            o_nodes (:obj: FactorNodes): a set of factor nodes

            o_id: the id of the factor node from o_nodes that will be at one
                end of the edge

            edge_type (o_nodes.EDGE_TYPE, optional): from the factor node's
                point-of-view, the type of edge this will be. Defaults to
                the 'default' edge type.
        """

        if edge_type is None:
            edge_type = 'default'

        assert edge_type in o_nodes.EDGE_TYPES, "Bad edge type: " + edge_type

        var_message_chunk = c_nodes.get_msg_chunk()

        #infer number of states in o_chunk_edge chunk
        c_chunk_num_states = var_message_chunk.num_states
        o_chunk_edge = o_nodes.message_chunks[edge_type]
        if o_chunk_edge.num_states == 0:
            o_chunk_edge.set_num_states(c_chunk_num_states)
        else:
            assert c_chunk_num_states == o_chunk_edge.num_states, \
                   "factor edge chunk and var chunk must have some number of states"

        self.graph_edge_info.add_edge(var_message_chunk, c_id, o_chunk_edge, o_id)

        #update maximum degree
        var_message_chunk.degree[c_id] += 1
        if var_message_chunk.max_degree < var_message_chunk.degree[c_id]:
            var_message_chunk.max_degree = var_message_chunk.degree[c_id]

        o_chunk_edge.degree[o_id] += 1
        if o_chunk_edge.max_degree < o_chunk_edge.degree[o_id]:
            o_chunk_edge.max_degree = o_chunk_edge.degree[o_id]

    def get_scheduled_nodes(self):
        """Get the Node instances in this factor graph scheduled for
        message-passing.
        """

        return self.nodes

    def finalize(self):
        """Prepares the factor graph for message-passing."""

        assert not self.__is_finalized, 'BP graph can only be finalized once.'
        assert len(self.nodes) > 0, 'No chunks added to message-passing schedule.' + \
                                     'Use add_chunk_to_schedule(chunk_obj) to add chunks'

        #finalize all the chunks we'll be passing messages for. needed for
        #cleanup and message setup.
        for chunk in self.nodes:
            chunk.finalize()

        self.graph_edge_info.finalize()
        self.__is_finalized = True

    def do_message_passing(self):
        """Performs the underlying message-passing algorithm (e.g., sum-product)
        on the factor graph.
        """

        assert self.__is_finalized, 'BP graph has not been finalized. Call ' + \
                                   'finalize() before message-passing.'

        self.prev_bel = [None]*len(self.nodes)

        for itt in range(0, self.bp_params['iters']):

            time0 = time.time()
            for chunk in self.nodes:
                msgs_hash = chunk.compute_messages()

                for key in msgs_hash.keys():
                    self._distribute_messages(chunk.message_chunks[key], msgs_hash[key])

            time1 = time.time()

            if itt != 0:
                (max_diff, is_converged) = self.__check_converged()
            else:
                max_diff, is_converged = 0, False

            print '%d: maxDiff: %f. Time: %f' %(itt, max_diff, time1-time0)

            if is_converged:
                self.streak_count += 1
            else:
                self.streak_count = 0

            if self.streak_count >= self.bp_params['streak_lim']:
                print "Converged on iteration: " + str(itt)
                break

            for i in range(0, len(self.nodes)):
                if isinstance(self.nodes[i], VarNodes):
                    self.prev_bel[i] = self.nodes[i].get_beliefs()

        for chunk in self.nodes:
            chunk.prepare_msgs_for_computation()

    def _distribute_messages(self, msg_chunk_source, msgs):
        """Distributes computed messages to their target nodes.

        Returns:
            msg_chunk_source (:obj: MessageChunk): the source MessageChunk that
                is sending the messages
            msgs (ndarray): messages to be sent
        """

        msgs = msgs.astype(float)

        msgs = MessageChunk.do_prepare_msgs_for_distribution(msgs)
        msg_chunk_dests = self.graph_edge_info.get_msg_chunk_dests(msg_chunk_source)
        for msg_dest in msg_chunk_dests:
            msg_dest.prepare_msgs_for_distribution()

            chunk_entries, to_index = msg_chunk_dests[msg_dest]
            source_idxs = chunk_entries[:, to_index[0]]
            dest_idxs = chunk_entries[:, to_index[1]]

            msg_dest.msgs_in[dest_idxs, :] *= self.bp_params['damp']
            msg_dest.msgs_in[dest_idxs, :] += (1-self.bp_params['damp'])*msgs[source_idxs, :]
            msg_dest.prepare_msgs_for_computation()

    def __check_converged(self):
        """Checks if the underyling message-passing algorithm has converged and
        returns statistics concerning convergence.

        Returns:
            For the VarNode instances in this factor graph, the maximum
            absolute difference in beliefs from this iteration and the last.
        """


        max_d_nodes = np.zeros(len(self.nodes))
        for i in range(0, len(self.nodes)):
            if isinstance(self.nodes[i], VarNodes):
                tmp = np.abs(self.prev_bel[i]-self.nodes[i].get_beliefs())
                max_d_nodes[i] = tmp.max()
            else:
                max_d_nodes[i] = 0

        max_diff = max_d_nodes.max()

        assert not math.isnan(max_diff)
        return (max_diff, max_diff <= self.bp_params['tol'])
