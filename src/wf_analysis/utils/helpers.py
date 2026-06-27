"""Generic utility functions."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(label: str = "Block") -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] Completed in {elapsed:.3f}s")


def chunk_list(lst: list, chunk_size: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
