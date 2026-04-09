"""Implementation of the ReduceMean operator."""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceMeanTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceMean operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceMean node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__ReduceMean.html
        # Only axis=-1 / axis=1 (reduce over the feature dimension) is supported.
        args = self._extract_reduce_args("ReduceMean")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            n = len(cols)
            if n == 0:
                raise ValueError(
                    "ReduceMean: input group must have at least one column."
                )
            mean_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                mean_expr = mean_expr + col
            mean_expr = mean_expr / ibis.literal(float(n))
            agg = self._optimizer.fold_operation(mean_expr)
            self._set_reduce_output(agg, keepdims)
        else:
            # Single column: mean of a scalar column is itself.
            self.set_output(data)
