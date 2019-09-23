import logging
from operator import itemgetter
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx

from utoolbox.stitching.error import NotConsolidatedError

__all__ = ["Sandbox"]

logger = logging.getLogger(__name__)


class Sandbox(object):
    def __init__(self, datastore):
        # populate nodes using datastore
        self._links = nx.Graph()
        logger.info("populating sandbox with datastore items...")
        shapes = []
        for name, image in datastore.items():
            logger.debug(f".. {name}")
            shape = image.shape
            self._links.add_node(name, shape=shape)
            shapes.append(shape)

        # determine initial shape by max image
        ndim = len(shapes[0])
        self._shape = tuple(
            max(shapes, key=itemgetter(idim))[idim] for idim in range(ndim)
        )
        logger.debug(f"initial shape {self.shape}")

        # only exists after consolidation
        self._sequence = None

    ##

    def link(self, reference, target, shift, similarity):
        """
        Add a link between nodes.
        
        Args:
            reference (str): reference image
            target (str): target image, the next reference image during fusion
            shift (tuple of float): translation of the target
            similarity (float): similarity between the two images, [0, 1]
        """
        self._links.add_edge(reference, target, shift=shift, weight=similarity)

    def update(self):
        self.preview()

    def preview(self):
        nx.draw(self._links)
        plt.show()

    def consolidate(self):
        """Fix edge weights into fusion sequence."""
        # build graph
        mst = nx.minimum_spanning_tree(self._links)

        # TODO determine bounding box
        for edge in mst.edges(data=True):
            pprint(edge)

        # TODO calculate shift list

    def result(self, tiles=None):
        """Fuse the result based on fusion sequence."""
        if self.sequence is None:
            raise NotConsolidatedError("please consolidate the sandbox first")

        # TODO determine overlaps

    ##

    @property
    def shape(self):
        """Estimated shape after fusion."""
        return self._shape

    @property
    def sequence(self):
        """Order to perform image fusions."""
        if self._sequence is None:
            raise RuntimeError("the sandbox is not consolidated yet")
        return self._sequence

