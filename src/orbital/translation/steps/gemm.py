"""Implementation of the Gemm (General Matrix Multiply) operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class GemmTranslator(Translator):
    """Translate the ONNX Gemm operator: ``C = alpha*A@B + beta*C``."""

    def process(self) -> None:
        """Translate the Gemm node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Gemm.html
        alpha = float(self._attributes.get("alpha", 1.0))
        beta = float(self._attributes.get("beta", 1.0))
        trans_a = int(self._attributes.get("transA", 0))
        trans_b = int(self._attributes.get("transB", 0))

        b_tensor = self._variables.get_initializer(self.inputs[1])
        if b_tensor is None:
            raise ValueError("Gemm: B (weight matrix) not found in initializers.")
        b_shape = list(b_tensor.dims)
        if len(b_shape) != 2:
            raise ValueError(f"Gemm: B must be 2D, got shape {b_shape}.")

        b_flat = self._variables.get_initializer_value(self.inputs[1])
        if b_flat is None or not isinstance(b_flat, (list, tuple)):
            raise ValueError("Gemm: B must be a constant list.")

        # C (bias) is optional per the ONNX spec.
        if len(self.inputs) < 3 or not self.inputs[2]:
            c_flat: list = [0.0]
        else:
            c_flat_val = self._variables.get_initializer_value(self.inputs[2])
            if c_flat_val is None or not isinstance(c_flat_val, (list, tuple)):
                raise ValueError("Gemm: C (bias) must be a constant list.")
            c_flat = list(c_flat_val)

        # Determine effective weight shape (input_dim, output_dim)
        if trans_b == 1:
            # B is stored as (output_dim, input_dim) → effective (input_dim, output_dim)
            output_dim, input_dim = b_shape

            def b_val(i: int, j: int) -> float:
                return b_flat[j * input_dim + i]  # B[j][i] → W[i][j]
        else:
            # B is stored as (input_dim, output_dim)
            input_dim, output_dim = b_shape

            def b_val(i: int, j: int) -> float:
                return b_flat[i * output_dim + j]  # W[i][j]

        first_operand = self._variables.consume(self.inputs[0])

        type_check = first_operand
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Gemm: A must be a numeric column or column group.")

        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            left_exprs = list(first_operand.values())
        else:
            left_exprs = [typing.cast(ibis.expr.types.NumericValue, first_operand)]

        num_features = len(left_exprs)
        if num_features != input_dim:
            raise ValueError(
                f"Gemm: A has {num_features} features but B expects {input_dim} inputs."
            )

        if len(c_flat) not in (output_dim, 1):
            raise ValueError(
                f"Gemm: C (bias) length {len(c_flat)} does not match output_dim {output_dim}."
            )

        results = []
        for j in range(output_dim):
            bias = c_flat[j] if len(c_flat) > 1 else c_flat[0]
            terms = [
                self._optimizer.fold_operation(left_exprs[i] * (alpha * b_val(i, j)))
                for i in range(num_features)
            ]
            output_j = sum(terms) + (beta * float(bias))
            results.append(self._optimizer.fold_operation(output_j))

        if output_dim == 1:
            self.set_output(results[0])
        else:
            self.set_output(
                ValueVariablesGroup({f"out_{j}": results[j] for j in range(output_dim)})
            )
