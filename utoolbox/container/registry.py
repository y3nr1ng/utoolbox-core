import logging
logger = logging.getLogger(__name__)

containers = {}

class ContainerRegistry(type):
    """Keep a record for all available data containers."""
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        containers[name] = cls
        logger.debug("New container \"{}\" added.".format(cls))
        return cls

class BaseContainer(metaclass=ContainerRegistry):
    pass
