"""Module for the Nodes class. See documentation for Nodes class."""

class Nodes(object):
    """This class represents a collection of nodes in a factor graph (stored as
    MessageChunks) and provides methods to perform operations on these
    MessageChunks. This class is meant to be abstract (without worrying what
    version of Python is being used). An implementation of the following
    functions must be given:

    _do_compute_messages

    Public methods:
        compute_messages: computes messages from the collection of nodes

        create_nodes: creates a given number of nodes

        finalize: prepares all contained nodes (and MessageChunks) for
            message-passing

        prepare_msgs_for_distribution: prepare nodes for message distribution

        prepare_msgs_for_computation: prepare nodes for message computation
    """

    def __init__(self, name='', nodes_params=None):
        """Initializer.

        Args:
            name (:obj:`str', optional): name of this Node object. Defaults to
                 the emptry string.

            nodes_params (dict, optional): parameters for this Node object needed for
                message-passing. Defaults to an empty dictionary.
        """

        self.name = name
        self.message_chunks = {}

        if nodes_params is None:
            self.nodes_params = {}
        else:
            self.nodes_params = nodes_params

        self.__finalized = False

    def compute_messages(self):
        """Function to perform message computation for all MessageChunks in this
        Nodes instance.

        Note: Messages of a MessageChunk are clamped to a minimum/maximum value,
            as indicated by the MessageChunk instance.

        Returns:
            A dictionary containing the computed messages.
        """

        msgs_dict = self._do_compute_messages()

        for key in msgs_dict.keys():
            msgs_dict[key] = self.message_chunks[key].clamp_messages(msgs_dict[key])

        return msgs_dict

    def create_nodes(self, num_to_create):
        """Creates nodes in all MessageChunks in this Nodes object.

        Args:
            num_to_create (int): number of new nodes to create.

        Returns:
            A list of ids to refer to created nodes.
        """

        assert not self.__finalized
        for i in self.message_chunks.keys():
            #all node ids should be the same
            node_ids = self.message_chunks[i].create_entries(num_to_create)
        return node_ids

    def finalize(self):
        """Prepares Nodes instance for message-passing.

        Preparation involves allocating space for messages and initializing
        messages for each Chunk contained in this Nodes instance.
        """

        assert not self.__finalized
        for key in self.message_chunks.keys():
            self.message_chunks[key].finalize()

        self.__finalized = True

    def prepare_msgs_for_distribution(self):
        """Prepare nodes for message distribution."""

        for key in self.message_chunks.keys():
            self.message_chunks[key].prepare_msgs_for_distribution()

    def prepare_msgs_for_computation(self):
        """Prepare nodes for message computation."""

        for key in self.message_chunks.keys():
            self.message_chunks[key].prepare_msgs_for_computation()

    def _do_compute_messages(self):
        """Helper function to compute messages. Must be overriden in a subclass.
        """

        raise RuntimeError('No implementation for _do_compute_messages given')
