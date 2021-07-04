import numpy as np


class BaseInstance(object):
    
    def __init__(self, n_node, feature):
        if not (n_node == feature.shape[0]):
            raise ValueError('   [Err] Parameters\' length are not match.'
                             'n_node: {}, feature: {}'.format(n_node,
                                                              feature.shape))
        self.n_node = n_node
        self._feature = feature
        self._node_list = np.arange(n_node, dtype=np.int)
        self._available_node = np.arange(1, n_node, dtype=np.int).tolist()
        self._adj = None
        self._x = None
        self._weights = None

        self.last_node = 0
        self.step_count = 0
        self._path = [self.last_node]
    
    def move(self, next_node, idx=None):
        pass

    def __len__(self):
        return self.n_node

    @property
    def weights(self):
        return self._weights
    
    @property
    def available_node(self):
        return self.available_node

    @property
    def adj(self):
        return self._adj
    
    @property
    def x(self):
        return self._x

    @property
    def path(self):
        return self._path
    
    @property
    def feature(self):
        return self._feature

    @property
    def node_list(self):
        return self._node_list
