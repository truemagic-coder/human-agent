"""Function calling system for HRM"""

from .registry import FunctionRegistry
from .builtin import register_builtin_functions

__all__ = ["FunctionRegistry", "register_builtin_functions"]
