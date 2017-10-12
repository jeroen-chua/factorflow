"""Module for the GraphEdgeInfo class. See documentation for GraphEdgeInfo class."""

import numpy as np

class GraphEdgeInfo(object):
    """This class represents the edges of a factor graph.

    Public methods:
        add_edge: adds an edge between a variable node and a factor node on a
            particular edge type.

        get_msg_chunk_dests: given a MessageChunk, returns the other
            MessageChunks passes messages to.

        finalize: prepares graph structure for message-passing.

    """

    __NODE_EDGE_BUFF = 1000

    def __init__(self):
        """Initializer."""

        self.edge_hash = {}
        self.edge_hash_count = {}
        self.to_chunks = {}

    def add_edge(self, c_msgs_chunk, c_id, o_msgs_chunk_edge, o_id):
        """Adds an edge between a variable node and a factor node on a
        particular edge type. This is accomplished by "connecting" their
        underlying message data structures.

        Args:
            c_msgs_chunk (:obj: MessageChunk): a message chunk representing
                the messages for a set of variable nodes.

            c_id: the id of the variable node from c_nodes that will be at one
                end of the edge.

            o_msgs_chunk_edge (:obj: MessageChunk): a message chunk representing
                the messages on the factor node's edge type.

            o_id: the id of the factor node that will be at one end of the edge.
        """

        c_nodes_loc = c_msgs_chunk.degree[c_id]
        o_msgs_edge_loc = o_msgs_chunk_edge.degree[o_id]

        key_chunk_pair = (c_msgs_chunk, o_msgs_chunk_edge)
        entry = [c_id, c_nodes_loc, o_id, o_msgs_edge_loc]

        if key_chunk_pair not in self.edge_hash.keys():
            self.edge_hash_count[key_chunk_pair] = 0
            self.edge_hash[key_chunk_pair] = tmp = \
                -1*np.ones((self.__NODE_EDGE_BUFF, 4), dtype='int')

        row_use = self.edge_hash_count[key_chunk_pair]
        if row_use > self.edge_hash[key_chunk_pair].shape[0]-1:
            tmp = -1*np.ones_like(self.edge_hash[key_chunk_pair])

            self.edge_hash[key_chunk_pair] = \
                np.concatenate((self.edge_hash[key_chunk_pair], tmp), axis=0)

        self.edge_hash[key_chunk_pair][row_use, :] = entry
        self.edge_hash_count[key_chunk_pair] += 1

    def get_msg_chunk_dests(self, msg_chunk):
        """Given a MessageChunk, returns the other MessageChunks it passes
        messages to.

        Args:
            msg_chunk (:obj:MessageChunk): the source MessageChunk.
        Returns:
            A dict where each key is a MessageChunk the source MessageChunk
            sends a message to. The key of the dict accesses a tuple. The
            first entry in the tuple is an Nx2 ndarray (call this ndarray
            chunk_entries) that stores the rows of the source/dest messages.
            The second entry in the tuple indicate which column in
            chunk_entries correspond to the source rows of the message
            structure, and which correspond to the destination rows of the
            message structure. A tuple of (0,1) indicates that column 0 is
            are the source indices and column 1 are the destination indices.
            A tuple of (1,0) indicates the opposite.
        """

        res = {} #key: chunk_idxs, [source_index,destination_index]

        if msg_chunk in self.to_chunks:

            list_of_dest_chunks = self.to_chunks[msg_chunk]
            for chunk_tuple in list_of_dest_chunks:
                chunk_entries = self.edge_hash[chunk_tuple]

                if msg_chunk == chunk_tuple[0]:
                    res[chunk_tuple[1]] = (chunk_entries, (0, 1))
                elif msg_chunk == chunk_tuple[1]:
                    res[chunk_tuple[0]] = (chunk_entries, (1, 0))
                else:
                    raise RuntimeError('Internal error: cannot find source message chunk?')

        return res
    def finalize(self):
        """Prepares graph structure for message-passing. In particular, removes
        excess pre-allocated buffers and prepares fancy-indexing operations
        needed for message distribution.
        """

        #remove buffered sizes
        for key in self.edge_hash.keys():
            self.edge_hash[key] = self.edge_hash[key][0:self.edge_hash_count[key]]

        for key in self.edge_hash.keys():
            chunk0 = key[0]
            chunk1 = key[1]

            tmp = self.edge_hash[key]
            new_idx = np.zeros((self.edge_hash[key].shape[0], 2), dtype='int')

            new_idx[:, 0] = tmp[:, 1]*chunk0.num_nodes
            new_idx[:, 0] += tmp[:, 0]

            new_idx[:, 1] = tmp[:, 3]*chunk1.num_nodes
            new_idx[:, 1] += tmp[:, 2]

            self.edge_hash[key] = new_idx
            for chunk in (chunk0, chunk1):
                if chunk in self.to_chunks:
                    self.to_chunks[chunk].append(key)
                else:
                    self.to_chunks[chunk] = [key]
