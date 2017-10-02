"""Module for the FactorNodes class. See documentation for FactorNodes class."""

from nodesLib import MessageChunk
from nodesLib.nodes import Nodes

class FactorNodes(Nodes):
    """This class represents a collection of factors in a factor graph, stored
    as multiple MessageChunks (one for each kind of edge associated with the
    factors) and provides methods to perform operations on these MessageChunks.
    Edge types are  defined by the Class field EDGE_TYPES, which can be
    overridden by a subclass. The purpose of edge-types is to allow different
    variable nodes to participate in different ways in the probability
    distribution. Eg, several variables might represent an input to a
    probabilistic-OR distribution, and another might represent the single 
    output of a probabilistic-OR.

    This class is an abstract subclass of the Nodes class. The following
    functions must be implemented by a concrete subclass:

    _do_compute_messages

    Public methods:
        get_msgs_on_edge: Returns the messages associated with a particular
        type of edge of the factors.


    """
    EDGE_TYPES = frozenset({'default'})

    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this FactorNodes object.
                 Defaults to the emptry string.

            nodes_params (dict, optional): parameters for this FactorNodes
                object needed for message-passing. Defaults to an empty
                dictionary.
        """

        super(FactorNodes, self).__init__(name, nodes_params)

        for i in self.EDGE_TYPES:
            chunk_name = self.name + '_' + i
            self.message_chunks[i] = MessageChunk(name=chunk_name, num_states=0)

    def get_msgs_on_edge(self, edge_type):
        """ Returns the messages associated with a particular type of edge of
        the factors.

        Args:
            edge_type (self.EDGE_TYPES): type of edge to retrieve

        Returns:
            The messages associated with the given edge_type
        """

        assert edge_type in self.EDGE_TYPES, 'Invalid edge type'
        return self.message_chunks[edge_type].msgs_in

