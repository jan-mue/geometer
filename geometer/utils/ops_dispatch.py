# Adapted from https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/ops_dispatch.pyx
# See license at https://github.com/pandas-dev/pandas/blob/main/LICENSE
from typing import Any, Literal

import numpy as np

DISPATCHED_UFUNCS = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "remainder",
    "matmul",
    "or",
    "xor",
    "and",
    "neg",
    "pos",
    "abs",
}
UNARY_UFUNCS = {
    "neg",
    "pos",
    "abs",
}
UFUNC_ALIASES = {
    "subtract": "sub",
    "multiply": "mul",
    "floor_divide": "floordiv",
    "true_divide": "truediv",
    "power": "pow",
    "remainder": "mod",
    "divide": "truediv",
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "bitwise_or": "or",
    "bitwise_and": "and",
    "bitwise_xor": "xor",
    "negative": "neg",
    "absolute": "abs",
    "positive": "pos",
}

# For op(., Array) -> Array.__r{op}__
REVERSED_NAMES = {
    "lt": "__gt__",
    "le": "__ge__",
    "gt": "__lt__",
    "ge": "__le__",
    "eq": "__eq__",
    "ne": "__ne__",
}


def maybe_dispatch_ufunc_to_dunder_op(
    obj: Any,
    ufunc: np.ufunc,
    method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
    *inputs: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch a ufunc to the equivalent dunder method.

    Args:
        obj: The object whose dunder method we dispatch to.
        ufunc: A NumPy ufunc.
        method: How the ufunc was called.
        inputs: The input arrays.
        kwargs: The additional keyword arguments, e.g. ``out``.

    Returns:
        The result of applying the ufunc
    """
    # special has the ufuncs we dispatch to the dunder op on

    op_name = ufunc.__name__
    op_name = UFUNC_ALIASES.get(op_name, op_name)

    def not_implemented(*args, **kwargs):  # type: ignore[no-untyped-def] # noqa: ARG001
        return NotImplemented

    if kwargs or ufunc.nin > 2:
        return NotImplemented

    if method == "__call__" and op_name in DISPATCHED_UFUNCS:
        if inputs[0] is obj:
            name = f"__{op_name}__"
            meth = getattr(obj, name, not_implemented)

            if op_name in UNARY_UFUNCS:
                if len(inputs) != 1:
                    raise ValueError(f"Unary ufunc {op_name} requires 1 input, got {len(inputs)}")
                return meth()

            return meth(inputs[1])

        elif inputs[1] is obj:
            name = REVERSED_NAMES.get(op_name, f"__r{op_name}__")

            meth = getattr(obj, name, not_implemented)
            result = meth(inputs[0])
            return result

        else:
            # should not be reached, but covering our bases
            return NotImplemented

    else:
        return NotImplemented
