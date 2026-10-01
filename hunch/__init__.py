"""Hunch: calibrated yes/no, pick-one and scale judgments from your own LLMs, read straight off the logprobs."""
__version__ = "1.6.2"

from .client import ajudge, judge  # noqa: E402,F401
from .engine import HunchError  # noqa: E402,F401
