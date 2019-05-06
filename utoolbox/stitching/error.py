class StitchingError(Exception):
    """Stitching module error base class."""

class NotConsolidatedError(StitchingError):
    """Sandbox is not consolidated yet, no fusion directive exists."""