"""Translator for the Cast operation."""

# CastLikeTranslator lives in its own module; re-exported here for compatibility.
from .castlike import CastLikeTranslator as CastLikeTranslator  # noqa: F401

import typing

import ibis
import onnx

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup

ONNX_TYPES_TO_IBIS: dict[int, ibis.expr.datatypes.DataType] = {
    onnx.TensorProto.FLOAT: ibis.expr.datatypes.float32,  # 1: FLOAT
    onnx.TensorProto.DOUBLE: ibis.expr.datatypes.float64,  # 11: DOUBLE
    onnx.TensorProto.STRING: ibis.expr.datatypes.string,  # 8: STRING
    onnx.TensorProto.INT64: ibis.expr.datatypes.int64,  # 7: INT64
    onnx.TensorProto.BOOL: ibis.expr.datatypes.boolean,  # 9: BOOL
}


class CastTranslator(Translator):
    """Processes a Cast node and updates the variables with the output expression.

    Cast operation is used to convert a variable from one type to another one
    provided by the attribute `to`.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Cast.html
        expr = self._variables.consume(self.inputs[0])
        to_type: int = typing.cast(int, self._attributes["to"])
        if to_type not in ONNX_TYPES_TO_IBIS:
            raise NotImplementedError(f"Cast: type {to_type} not supported")

        target_type = ONNX_TYPES_TO_IBIS[to_type]

        def _is_numeric_or_bool(value: ibis.Value) -> bool:
            vtype = value.type()
            return (
                hasattr(vtype, "is_numeric")
                and vtype.is_numeric()
                or hasattr(vtype, "is_boolean")
                and vtype.is_boolean()
            )

        if (
            self._options.allow_text_tensors is False
            and target_type == ibis.expr.datatypes.string
        ):
            # When sklearn2onnx needs to concatenate features into a single tensor
            # it homogenizes their dtype (e.g. casts numeric target-encoder output to
            # string so it can sit alongside passthrough string columns). Besides being
            # redundant for SQL consumers, promoting one column to text forces the whole
            # encoded block to become text as well, so we drop it unless the caller
            # explicitly opts in.
            if isinstance(expr, VariablesGroup):
                if all(_is_numeric_or_bool(expr.as_value(k)) for k in expr):
                    self.set_output(expr)
                    return
            elif isinstance(expr, ibis.Value) and _is_numeric_or_bool(expr):
                self.set_output(expr)
                return

        if isinstance(expr, VariablesGroup):
            casted = ValueVariablesGroup(
                {
                    k: self._optimizer.fold_cast(expr.as_value(k).cast(target_type))
                    for k in expr
                }
            )
            self.set_output(casted)
        elif isinstance(expr, ibis.Value):
            self.set_output(self._optimizer.fold_cast(expr.cast(target_type)))
        else:
            raise ValueError(
                f"Cast: expected a column group or a single column. Got {type(expr)}"
            )


