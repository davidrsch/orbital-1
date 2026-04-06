"""Implementation of the Erf operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup

# Abramowitz & Stegun (1964) §7.1.28 rational approximation for erf(x).
# Max absolute error: |erf(x) - approx(x)| < 1.5e-7 for all real x.
# Formula: erf(x) ≈ sign(x) * (1 - poly(t) * exp(-x²))
# where t = 1 / (1 + 0.3275911 * |x|),  poly(t) = sum(a_i * t^i, i=1..5)
#
# NOTE: This is an intentional approximation. The ONNX Erf operator specifies
# the exact mathematical erf() function, but most SQL dialects — including
# DuckDB, SQLite, and Spark SQL — do not expose erf() as a native scalar
# function.  The A&S polynomial achieves float32-level accuracy (max error
# ≈ 1.5e-7) which is sufficient for all practical ANN inference use cases.
# The same polynomial is used by orbital's R counterpart (ann-helpers.R
# .erf_approx_expr) and by the Gelu translator in approximate=None mode.
_P = 0.3275911
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429


def _erf_approx(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Compute erf(v) via the A&S 7.1.28 polynomial approximation (max error ~1.5e-7)."""
    ax = v.abs()
    sign = ibis.cases(
        (v >= ibis.literal(0.0), ibis.literal(1.0)),
        else_=ibis.literal(-1.0),
    )
    t = ibis.literal(1.0) / (ibis.literal(1.0) + ibis.literal(_P) * ax)
    # Horner evaluation: t * (a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    poly = t * (
        ibis.literal(_A1)
        + t
        * (
            ibis.literal(_A2)
            + t * (ibis.literal(_A3) + t * (ibis.literal(_A4) + t * ibis.literal(_A5)))
        )
    )
    return sign * (ibis.literal(1.0) - poly * (ibis.literal(-1.0) * ax * ax).exp())


class ErfTranslator(Translator):
    """Translate the ONNX Erf operator using a polynomial approximation."""

    def process(self) -> None:
        """Translate the Erf node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Erf.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Erf: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {
                    k: self._optimizer.fold_operation(_erf_approx(v))
                    for k, v in data.items()
                }
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_erf_approx(data))

        self.set_output(result)
