"""Windowed block reduction."""

from .reducer import (
    BlockReducer,
    BlockReduction,
    ProductTerm,
    RNEAlignmentPolicy,
    WindowOverflowError,
    container_bounds,
)

__all__ = [
    "BlockReducer",
    "BlockReduction",
    "ProductTerm",
    "RNEAlignmentPolicy",
    "WindowOverflowError",
    "container_bounds",
]
