"""Rectangular IFS parser and matrix systems."""

from .matrix import DeterministicFunctionSystem, FunctionSystem, FunctionSystemRandom
from .parser import ifs, interpret_file

__all__ = [
    "DeterministicFunctionSystem",
    "FunctionSystem",
    "FunctionSystemRandom",
    "ifs",
    "interpret_file",
]
