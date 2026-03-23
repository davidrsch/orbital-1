"""Implementation of the LogSoftmax operator."""

import functools
import operator

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class LogSoftmaxTranslator(Translator):
    """Computes log(softmax(x)) with numerical stability via the log-sum-exp trick.

    For a group of columns x_1 … x_n::

        log_softmax(x_i) = x_i - log(∑ exp(x_j))

    Numerically stable form (subtracts max before exponentiating)::

        log_softmax(x_i) = (x_i − max_x) − log(∑ exp(x_j − max_x))

    For a single scalar input the result is always 0.0 (softmax of a singleton = 1).
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx__LogSoftmax.html
        data = self._variables.consume(self.inputs[0])

        axis = self._attributes.get("axis", -1)
        if axis not in (-1, 1):
            raise ValueError(
                "LogSoftmax: only axis=-1 or axis=1 are supported in the SQL context."
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            values = list(data.values())

            # Running max for numerical stability
            max_x: ibis.expr.types.NumericValue = values[0]
            for v in values[1:]:
                max_x = ibis.greatest(max_x, v)

            # exp(x_i − max_x) for each column
            shifted_exps = [(v - max_x).exp() for v in values]

            # log(∑ exp(x_j − max_x))
            sum_exp = functools.reduce(operator.add, shifted_exps)
            log_sum = sum_exp.log()

            # log_softmax(x_i) = (x_i − max_x) − log_sum
            result = {k: (v - max_x) - log_sum for k, v in zip(data.keys(), values)}
            self.set_output(ValueVariablesGroup(result))

        elif isinstance(data, ibis.expr.types.NumericValue):
            # Single value: log(softmax(x)) = log(1) = 0
            self.set_output(ibis.literal(0.0).cast("float64"))

        else:
            raise ValueError(
                f"LogSoftmax: expected a numeric value or a column group, got {type(data)}"
            )
