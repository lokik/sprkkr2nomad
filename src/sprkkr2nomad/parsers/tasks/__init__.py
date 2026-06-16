"""Per-task NOMAD section builders for SPRKKR output files.

Importing this package registers all built-in task parsers (scf, dos, bsf) via
``SprkkrTaskParser.__init_subclass__``.
"""

from .base import SprkkrTaskParser
from . import scf, dos, bsf  # noqa: F401 — trigger subclass registration

__all__ = ['SprkkrTaskParser']
