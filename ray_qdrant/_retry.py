"""Retries for transient Qdrant errors.

Based on ``ray._common.retry.call_with_retry``, which is internal to Ray.
"""

import logging
import random
import re
import time
import traceback
from collections.abc import Callable, Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

R = TypeVar("R")

_MAX_BACKOFF_S = 32


def call_with_retry(
    f: Callable[[], R],
    description: str,
    match: Sequence[str],
    max_attempts: int,
) -> R:
    """Call ``f``, retrying with exponential backoff on matching errors.

    Args:
        f: The function to call.
        description: What ``f`` does, for logging, like "create collection 'x'".
        match: Regexes searched for in ``"ClassName: message"`` of the raised
            exception. Other errors are raised immediately.
        max_attempts: The maximum number of attempts.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return f()
        except Exception as e:
            error = "".join(traceback.format_exception_only(type(e), e)).rstrip()
            if attempt == max_attempts or not any(re.search(p, error) for p in match):
                raise
            # Exponential backoff with 20% jitter.
            backoff = min(2 ** (attempt - 1), _MAX_BACKOFF_S) * random.uniform(0.8, 1.2)
            logger.debug(
                "Retrying %s after %.2fs (attempt %d): %s",
                description,
                backoff,
                attempt,
                error,
            )
            time.sleep(backoff)
    raise AssertionError("unreachable")
