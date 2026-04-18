"""Canonical default values for ONNX activation operators.

These constants centralise the small numeric defaults duplicated across
individual operator implementations.  Keeping them in one place avoids silent
drift if the ONNX spec ever changes, and makes it trivial to audit the values.
"""

# ONNX defaults (see https://onnx.ai/onnx/operators/).
ELU_ALPHA: float = 1.0
"""Default alpha for the ONNX ``Elu`` operator."""

LEAKYRELU_ALPHA: float = 0.01
"""Default alpha for the ONNX ``LeakyRelu`` operator."""

CELU_ALPHA: float = 1.0
"""Default alpha for the ONNX ``Celu`` operator."""
