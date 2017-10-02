"""Module for the MessageChunk class."""

import numpy as np
from enum import Enum

class MessageChunk(object):
    """This class represents a group of messages in the factor graph and
    provides an interface to perform operations with these messages.

    The key pieces of information this class stores is:
        1) a group of messages (stored in msgs_in), and
        2) any message-chunk specific parameters

    Public methods:
        create_entries: create entries in this MessageChunk

        finalize: prepare MessageChunk for message-passing

        set_num_states: set the number of states for the message entities.

        clamp_messages: clamps a given message to a range of values, before
            normalization

        do_prepare_msgs_for_distribution: static method to prepare a message
            data structure for distribution to other MessageChunks

        prepare_msgs_for_distribution: prepare messages for message distribution.

        prepare_msgs_for_computation: prepare messages for message computation.
    """

    #Strategies to initialize messages. Either randomly or uniformly.
    MSGS_INIT_ENUM = Enum('msgs_init_enum', 'random uniform')

    __message_chunk_count = 0
    _MSG_MIN_VAL = 1e-8 #minimum value a message may have
    _MSG_MAX_VAL = 1.0-1e-8 #maximum value a message may have

    #used to facilitate creation of buffers to speed-up MessageChunk creation.
    __NODE_BUFF_SZ = 10000

    def __init__(self, name='', num_states=2):
        """Initializer.

        Args:
            name (:obj:`str', optional): An identifier (name) for this MessageChunk.
                 Defaults to the empty string.

            num_states (int): For a MessageChunk representing variable nodes,
                this is the number of states these variables have. For factor
                nodes,  this is the number of states the variable nodes
                connected to this MessageChunk have. Defaults to 2.
        """
        self.__msgs_init_strat = self.MSGS_INIT_ENUM.random
        self.__num_entries = 0

        self.name = name
        self.num_states = num_states
        self.__message_chunk_id = MessageChunk.__message_chunk_count
        MessageChunk.__message_chunk_count += 1

        self.max_degree = 0

        self.__msgs_init_range = 0.2
        self.__msgs_init_min = 0.4

        self.pad_msg_val = []

        #size: [#degree,#states,#nodes]. keeps track of incoming messages
        self.msgs_in = np.zeros([0, 0, 0])

        #keeps track of node's current degree
        self.degree = np.zeros([self.__NODE_BUFF_SZ], dtype='int')
        self.__is_rolled = True
        self._finalized = False

    def create_entries(self, num_entries_to_create):
        """Creates entries in this MessageChunk.

        Args:
            num_entries_to_create (int): number of new entries to create.

        Returns:
            A list of ids to refer to the nodes just created.
        """

        assert not self._finalized, \
        'Cannot make changes to a finalized MessageChunk'

        node_ids = range(self.__num_entries, self.__num_entries+num_entries_to_create)
        self.__num_entries += num_entries_to_create

        while self.degree.size < self.__num_entries:
            tmp = np.zeros_like(self.degree, dtype='int')
            self.degree = np.concatenate((self.degree, tmp), axis=0)

        return node_ids

    def finalize(self):
        """Prepares MessageChunk for message-passing.

        Preparation involves allocating space for messages and initializing
        messages.
        """

        assert not self._finalized, \
        'MessageChunk: %s is already finalized' %(self.name)

        assert self.__num_entries > 0, \
        'MessageChunk: %s is empty' %(self.name)

        self.degree = self.degree[0:self.__num_entries]

        self.msgs_in = self.__alloc_message()

        if self.pad_msg_val == []:
            self.pad_msg_val = 1.0/self.num_states

        pad_msg_val = self.pad_msg_val*np.ones(self.num_states)

        for i in range(0, self.msgs_in.shape[2]):
            for j in range(0, self.num_states):
                self.msgs_in[self.degree[i]:, j, i] = pad_msg_val[j]

        self._finalized = True

    def set_num_states(self, num_states):
        """Set the number of states for the message entities.

        Args:
            num_states (int): number of states.
        """

        assert num_states > 0, 'Number of states must be > 0'

        assert not self._finalized, \
        'Cannot make changes to a finalized MessageChunk'

        self.num_states = num_states

    def __alloc_message(self):
        """Allocates the message data structure (msgs_in).

        Returns:
            The initialized messages as an ndarray
        """

        sz_msg = [self.max_degree, self.num_states, self.__num_entries]

        if np.prod(sz_msg) == 0:
            return np.random.random_sample(size=sz_msg)

        if self.msgs_init_strat == self.MSGS_INIT_ENUM.uniform:
            msg_range = 0.0
            msg_min = 1.0
        elif self.msgs_init_strat == self.MSGS_INIT_ENUM.random:
            if isinstance(self.msgs_init_range, int) or isinstance(self.msgs_init_range, float):

                msg_range = self.msgs_init_range
            else:
                msg_range = np.asarray(self.msg_range)
                msg_range = np.reshape(msg_range, (1, msg_range.size, 1))

            if isinstance(self.msgs_init_min, int) or isinstance(self.msgs_init_min, float):
                msg_min = self.msgs_init_min
            else:
                msg_min = np.asarray(self.msg_min)
                msg_min = np.reshape(msg_min, (1, msg_min.size, 1))

        return msg_min + msg_range*np.random.random_sample(size=sz_msg)

    def clamp_messages(self, msg):
        """Clamp the given messages to be in the range
        [self._MSG_MIN_VAL, self._MSG_MAX_VAL], and then normalize them to sum
        to 1 across axis 1.

        Args:
            msg (ndarray): ndarray of doubles to be clamped

        Returns:
            The clamped and normalized messages
        """

        changed = False
        if msg.size > 0:
            if np.max(msg) > self._MSG_MAX_VAL:
                msg[msg > self._MSG_MAX_VAL] = self._MSG_MAX_VAL
                changed = True
            if np.min(msg) < self._MSG_MIN_VAL:
                msg[msg < self._MSG_MIN_VAL] = self._MSG_MIN_VAL
                changed = True
        if changed:
            msg /= np.sum(msg, axis=1, keepdims=True)
        return msg

    @staticmethod
    def do_prepare_msgs_for_distribution(msgs):
        """Reorder and reshape messages to allow for distribution of messages
        during message-passing. If msgs_sz=msgs.shape, then msgs has axes 1 and
        2 swapped, then reshaped to be of size
        [msgs_sz[0]*msgs_sz[2], msgs_sz[1]].

        Args:
            msgs (ndarray): 3D array of doubles to be reordered/reshaped.

        Returns:
            The reordered/reshaped messages.
        """

        assert msgs.ndim == 3, 'msgs.ndim must be 3'

        msgs_sz = msgs.shape
        msgs = np.swapaxes(msgs, 1, 2)
        msgs = np.reshape(msgs, [msgs_sz[0]*msgs_sz[2], msgs_sz[1]])

        return msgs

    def prepare_msgs_for_distribution(self):
        """Prepare messages for message distribution.

        If self.msgs_in is not yet prepared for message distribution, then
        msg_sz=self.msgs_in.shape, then self.msgs_in has axes 1 and 2 swapped,
        then reshaped to be of size [msg_sz[0]*msg_sz[2], msg_sz[1]]. This is
        the inverse of self.prepare_msgs_for_computation().
        """

        assert self._finalized, 'Cannot unroll. MessageChunk must be finalized'

        if self.__is_rolled:
            self.msgs_in = MessageChunk.do_prepare_msgs_for_distribution(self.msgs_in)
            self.__is_rolled = False

    def prepare_msgs_for_computation(self):
        """Prepare messages for message computation.

        If self.msgs_in is not yet prepared for message computation, then
        msg_sz=self.msgs_in.shape, then self.msgs_in is reshaped to be of size
        [self.max_degree, self.__num_entries, self.num_states] and then has axes
        1 and 2 swapped. This is the inverse of self.prepare_msgs_for_distribution().
        """

        assert self._finalized, 'Cannot roll. MessageChunk must be finalized'

        if not self.__is_rolled:
            self.msgs_in = np.reshape(self.msgs_in, [self.max_degree, \
                                                     self.__num_entries, \
                                                     self.num_states] \
                                     )
            self.msgs_in = np.swapaxes(self.msgs_in, 1, 2)
            self.__is_rolled = True

    @property
    def message_chunk_id(self):
        """ Get message_chunk_id. Guaranteed to be unique. """

        return self.__message_chunk_id

    @property
    def num_nodes(self):
        """ Get the number of nodes in this MessageChunk. """

        return self.__num_entries

    @property
    def msgs_init_min(self):
        """ Get the minimum value for message initialization """

        return self.__msgs_init_min

    @msgs_init_min.setter
    def msgs_init_min(self, val):
        assert val >= 0, 'msg_init_min must be >= 0'
        self.__msgs_init_min = val

    @property
    def msgs_init_range(self):
        """ Get the range of values for message initialization """

        return self.__msgs_init_range

    @msgs_init_range.setter
    def msgs_init_range(self, vals):
        self.__msgs_init_range = vals

    @property
    def msgs_init_strat(self):
        """ Get the message initialization strategy. For valid values, see
            MessageChunk.MSGS_INIT_ENUM"""

        return self.__msgs_init_strat

    @msgs_init_strat.setter
    def msgs_init_strat(self, strat):
        self.__msgs_init_strat = strat
