"""Implementation of the ScatterElements operator (ONNX opset 11+).

ScatterElements performs scatter updates on individual elements of a tensor,
writing ``updates`` values to positions given by ``indices``.

Limitations
-----------
* Only ``axis=0`` (the default) is supported.
* ``indices`` and ``updates`` must be constant initializers (not dynamic
  layer outputs).
* Supported ``reduction`` values: ``"none"`` (default), ``"add"``, ``"mul"``,
  ``"min"``, ``"max"``.

References
----------
https://onnx.ai/onnx/operators/onnx__ScatterElements.html
"""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ScatterElementsTranslator(Translator):
    """Translate the ONNX ScatterElements operator for 1-D axis=0 scatter."""

    def process(self) -> None:
        """Translate the ScatterElements node, writing result to the graph."""
        axis = int(self._attributes.get("axis", 0))
        reduction = str(self._attributes.get("reduction", "none"))

        if axis != 0:
            raise NotImplementedError(
                f"ScatterElementsTranslator: axis={axis} is not supported. "
                "Only axis=0 (the default) can be expressed in SQL. "
                "ScatterElements with arbitrary axis requires mutable tensor indexing "
                "which is not available in SQL. "
                "If you need this, please open an issue at "
                "https://github.com/posit-dev/orbital/issues."
            )

        # indices and updates must be constant initializers
        indices_val = self._variables.get_initializer_value(self.inputs[1])
        updates_val = self._variables.get_initializer_value(self.inputs[2])
        if indices_val is None or updates_val is None:
            raise NotImplementedError(
                "ScatterElementsTranslator: indices and updates must be constant initializers"
            )

        # Normalise indices and updates to flat Python lists
        if isinstance(indices_val, (int, float)):
            indices = [int(indices_val)]
        else:
            indices = [int(x) for x in indices_val]

        if isinstance(updates_val, (int, float)):
            updates = [float(updates_val)]
        else:
            updates = [float(x) for x in updates_val]

        # Consume data input (dynamic layer output or constant)
        data = self._variables.consume(self.inputs[0])
        if isinstance(data, VariablesGroup):
            data_exprs: list[ibis.expr.types.NumericValue] = list(data.values())
        elif isinstance(data, (list, tuple)):
            data_exprs = [ibis.literal(float(x)) for x in data]
        else:
            data_exprs = [data]  # type: ignore[list-item]

        n = len(data_exprs)
        results: list[ibis.expr.types.NumericValue] = []

        for i in range(n):
            matching = [j for j, idx in enumerate(indices) if idx == i]

            if not matching:
                results.append(data_exprs[i])
                continue

            if reduction == "none":
                # Last-write wins for duplicate indices
                results.append(
                    self._optimizer.fold_operation(ibis.literal(updates[matching[-1]]))
                )
            elif reduction == "add":
                upd_sum = sum(updates[j] for j in matching)
                results.append(
                    self._optimizer.fold_operation(
                        data_exprs[i] + ibis.literal(upd_sum)
                    )
                )
            elif reduction == "mul":
                upd_prod = 1.0
                for j in matching:
                    upd_prod *= updates[j]
                results.append(
                    self._optimizer.fold_operation(
                        data_exprs[i] * ibis.literal(upd_prod)
                    )
                )
            elif reduction == "min":
                upd_min = min(updates[j] for j in matching)
                results.append(
                    self._optimizer.fold_operation(
                        ibis.least(data_exprs[i], ibis.literal(upd_min))
                    )
                )
            elif reduction == "max":
                upd_max = max(updates[j] for j in matching)
                results.append(
                    self._optimizer.fold_operation(
                        ibis.greatest(data_exprs[i], ibis.literal(upd_max))
                    )
                )
            else:
                raise NotImplementedError(
                    f"ScatterElementsTranslator: unsupported reduction '{reduction}'"
                )

        if len(results) == 1:
            self.set_output(results[0])
        else:
            self.set_output(
                ValueVariablesGroup(
                    {f"out_{i}": results[i] for i in range(len(results))}
                )
            )
