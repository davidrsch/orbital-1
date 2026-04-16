"""Implementation of the BatchNormalization operator."""

import typing

import ibis

from ..variables import NumericVariablesGroup, VariablesGroup
from ._base_norm import NormTranslatorBase


class BatchNormalizationTranslator(NormTranslatorBase):
    """Translate the ONNX BatchNormalization operator (inference mode)."""

    def process(self) -> None:
        """Translate the BatchNormalization node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
        # Inference mode only (5 inputs: X, scale γ, bias β, running_mean μ, running_var σ²)
        # y_i = ((x_i - mean_i) / sqrt(var_i + eps)) * scale_i + bias_i
        data = self._variables.consume(self.inputs[0])
        scale, bias, epsilon = self._extract_scale_bias_epsilon(
            op_name="BatchNormalization"
        )

        if bias is None:
            raise ValueError("BatchNormalization: bias must be a constant initializer.")

        mean = self._variables.get_initializer_value(self.inputs[3])
        var = self._variables.get_initializer_value(self.inputs[4])

        if not isinstance(mean, (list, tuple)):
            raise ValueError("BatchNormalization: mean must be a constant initializer.")
        if not isinstance(var, (list, tuple)):
            raise ValueError("BatchNormalization: var must be a constant initializer.")

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("BatchNormalization: input must be a numeric column.")

        def _bn(
            v: ibis.expr.types.NumericValue,
            s: float,
            b: float,
            m: float,
            vr: float,
        ) -> ibis.expr.types.NumericValue:
            std = (vr + epsilon) ** 0.5
            return (v - ibis.literal(m)) / ibis.literal(std) * ibis.literal(
                s
            ) + ibis.literal(b)

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            fields = list(data.keys())
            if len(fields) != len(scale):
                raise ValueError(
                    f"BatchNormalization: number of input features ({len(fields)}) "
                    f"must match scale length ({len(scale)})."
                )
            result = NumericVariablesGroup(
                {
                    field: self._optimizer.fold_operation(
                        _bn(
                            data[field],
                            float(scale[i]),
                            float(bias[i]),
                            float(mean[i]),
                            float(var[i]),
                        )
                    )
                    for i, field in enumerate(fields)
                }
            )
        else:
            if len(scale) != 1:
                raise ValueError(
                    "BatchNormalization: expected single-element parameters for a single-column input."
                )
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(
                _bn(
                    data, float(scale[0]), float(bias[0]), float(mean[0]), float(var[0])
                )
            )

        self.set_output(result)
