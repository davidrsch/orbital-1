"""Shared base class for ONNX normalisation translators."""

import ibis

from ..translator import Translator


class NormTranslatorBase(Translator):
    """Mixin providing scale, bias, and epsilon extraction for normalisation operators.

    Subclasses call :meth:`_extract_scale_bias_epsilon` to retrieve the standard
    affine parameters and epsilon hyperparameter shared by BatchNormalization,
    LayerNormalization, InstanceNormalization, GroupNormalization, and
    RMSNormalization.
    """

    def _extract_scale_bias_epsilon(
        self,
        scale_idx: int = 1,
        bias_idx: int | None = 2,
        op_name: str | None = None,
        epsilon_default: float = 1e-5,
    ) -> tuple:
        """Extract scale, optional bias, and epsilon from the ONNX node inputs.

        Parameters
        ----------
        scale_idx:
            Input index for the scale (gamma) tensor (default 1).
        bias_idx:
            Input index for the bias (beta) tensor, or ``None`` if the operator
            has no bias input (e.g. RMSNormalization).
        op_name:
            Human-readable operator name used in error messages.  Defaults to
            the class name.

        Returns
        -------
        tuple[list, list | None, float]
            ``(scale, bias, epsilon)`` — ``bias`` is ``None`` when ``bias_idx``
            is ``None`` or the corresponding input is absent.

        Raises
        ------
        ValueError
            If scale is not a constant initializer, or if a present bias is not
            a constant initializer.
        """
        name = op_name or type(self).__name__
        epsilon = float(self._attributes.get("epsilon", epsilon_default))

        scale = self._variables.get_initializer_value(self.inputs[scale_idx])
        if not isinstance(scale, (list, tuple)):
            raise ValueError(
                f"{name}: scale (inputs[{scale_idx}]) must be a constant initializer."
            )

        bias: list | None = None
        if bias_idx is not None and len(self.inputs) > bias_idx:
            bias_raw = self._variables.get_initializer_value(self.inputs[bias_idx])
            if bias_raw is not None:
                if not isinstance(bias_raw, (list, tuple)):
                    raise ValueError(
                        f"{name}: bias (inputs[{bias_idx}]) must be a constant initializer."
                    )
                bias = bias_raw

        return scale, bias, epsilon

    def _compute_mean_std(self, cols: list, n: int, epsilon: float):
        """Compute per-row mean and std expressions from feature column expressions.

        Used by LayerNormalization and InstanceNormalization which share the
        same per-row normalisation formula.

        Parameters
        ----------
        cols:
            List of ibis column expressions (one per feature/channel).
        n:
            Number of columns (same as ``len(cols)``).
        epsilon:
            Small constant added inside the square-root to avoid division by zero.

        Returns
        -------
        tuple[ibis_expr, ibis_expr]
            ``(mean_expr, std_expr)`` — both are inline SQL expressions
            evaluated per row.
        """
        mean_expr = cols[0]
        for col in cols[1:]:
            mean_expr = mean_expr + col
        mean_expr = mean_expr / ibis.literal(float(n))

        var_terms = [(col - mean_expr) ** 2 for col in cols]
        var_expr = var_terms[0]
        for t in var_terms[1:]:
            var_expr = var_expr + t
        var_expr = var_expr / ibis.literal(float(n))

        std_expr = (var_expr + ibis.literal(epsilon)) ** ibis.literal(0.5)
        return mean_expr, std_expr
