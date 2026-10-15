"""Thin re-export of :class:`critic_sharing.utils.seeding.SeedManager`.

The actual implementation lives inside the installable package so the
CLI (``critic-sharing eval`` etc.) can import it without
needing the repo's ``evaluations/`` tree on PYTHONPATH. This module
exists so analyze.py / run.py files in ``evaluations/`` can keep a
short import path:

    from evaluations._common.seeding import SeedManager
"""

from critic_sharing.utils.seeding import SeedManager  # noqa: F401

__all__ = ["SeedManager"]
