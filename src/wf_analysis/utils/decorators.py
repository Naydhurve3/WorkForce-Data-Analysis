"""Decorators for logging, timing, and caching."""

import functools
import hashlib
import json
import pickle
import time
from pathlib import Path

from loguru import logger


def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Finished {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


def cache_to_disk(cache_dir: str = ".cache"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            key = hashlib.md5(
                json.dumps(
                    {
                        "args": str(args),
                        "kwargs": str(sorted(kwargs.items())),
                    },
                    default=str,
                ).encode()
            ).hexdigest()

            cache_file = cache_path / f"{func.__name__}_{key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper

    return decorator
