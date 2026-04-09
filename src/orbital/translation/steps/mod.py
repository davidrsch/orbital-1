"""Implementation of the Mod operator (ONNX opset 10+).

Element-wise modulo:

    fmod=0 (default): y = x - trunc(x / y_divisor) * y_divisor  (C-style %)
    fmod=1:           y = fmod(x, y_divisor)  (C fmod, sign matches dividend)

References
----------
https://onnx.ai/onnx/operators/onnx__Mod.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ModTranslator(Translator):
    """Translate the ONNX Mod operator (element-wise modulo)."""

    def process(self) -> None:
        """Translate the Mod node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        divisor_val = self._variables.get_initializer_value(self.inputs[1])

        if divisor_val is None:
            raise NotImplementedError(
                "Mod: divisor (inputs[1]) must be a constant initializer."
            )

        fmod = int(self._attributes.get("fmod", 0))

        if isinstance(divisor_val, (list, tuple)):
            if len(divisor_val) != 1:
                raise NotImplementedError(
                    "Mod: only scalar divisor is supported in the tabular context."
                )
            divisor = float(divisor_val[0])
        else:
            divisor = float(divisor_val)

        d_lit = ibis.literal(divisor)

        def _mod(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            if fmod:
                # C fmod: remainder has sign of dividend
                rem = v - (v / d_lit).cast("int64").cast("float64") * d_lit
            else:
                # floor modulo (sign follows divisor), as required by ONNX fmod=0
                rem = v % d_lit
            return rem

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_mod(v)) for k, v in data.items()}
            )
        else:
            result = self._optimizer.fold_operation(_mod(data))

        self.set_output(result)
