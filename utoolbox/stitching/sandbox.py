import logging

import networkx as nx

__all__ = [
    'Sandbox'
]

logger = logging.getLogger(__name__)

class Sandbox(object):
    def __init__(self, ds):
        self._datastore = ds
        self._links = nx.Graph()

        # populate the graph
        self._links.add_nodes_from(ds.keys())

    @property
    def shape(self):
        pass

    @property
    def datastore(self):
        return self._datastore

    def link(self, ref, target, shift, weight=1.):
        self._links.add_edge(
            ref, target, 
            weight=weight, shift=shift
        )
    
    def consolidate(self):
        # build graph
        mst = nx.minimum_spanning_tree(self._links)

        #TODO determine bounding box

        #TODO calculate shift list

    def result(self, tiles=None):
        #TODO determine overlaps

        pass
