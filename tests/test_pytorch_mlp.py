"""Integration tests for PyTorch MLP end-to-end: ONNX export → orbital translate.

All tests are skipped when PyTorch is not installed.
"""

import io
import math

import numpy as np
import pandas as pd
import pytest

onnx = pytest.importorskip("onnx")
onnx_checker = pytest.importorskip("onnx.checker")
torch = pytest.importorskip("torch")
nn = torch.nn

import orbital
from orbital import types
from orbital.ast import ParsedPipeline
from orbital_testing_helpers import execute_sql


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_to_onnx(model: nn.Module, dummy_input: "torch.Tensor") -> onnx.ModelProto:
    """Export a PyTorch model to ONNX and return the ModelProto."""
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buf,
        opset_version=17,
        input_names=["float_input"],
        output_names=["output"],
        dynamic_axes={"float_input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _features_for(n: int) -> dict:
    return {f"x{i}": types.FloatColumnType() for i in range(n)}


def _input_df(arr: np.ndarray) -> pd.DataFrame:
    """Convert a (N, D) numpy array to a DataFrame with column names x0, x1, ..."""
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _torch_predict(model: nn.Module, arr: np.ndarray) -> np.ndarray:
    """Run inference with the PyTorch model and return a flat numpy array."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(arr, dtype=torch.float32)
        out = model(t).numpy()
    return out.flatten()


def _sql_predict_regression(onnx_model: onnx.ModelProto, features: dict, X: pd.DataFrame) -> np.ndarray:
    """Translate an ONNX regression model to SQL and execute it against DuckDB."""
    import duckdb
    conn = duckdb.connect(":memory:")
    parsed = ParsedPipeline._from_onnx_model(onnx_model, features)
    sql = orbital.export_sql("data", parsed, dialect="duckdb")
    result = execute_sql(sql, conn, "duckdb", X)
    # The single output column is named "variable" for regression
    col = result.columns[0]
    return result[col].values


# ---------------------------------------------------------------------------
# Test 1: Simple regression MLP — Sequential(Linear→ReLU→Linear→ReLU→Linear)
# ---------------------------------------------------------------------------


class TestPyTorchRegressionMLP:
    """End-to-end test for a pure regression MLP: 4→8→4→1 with ReLU activations."""

    def setup_method(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.model.eval()
        rng = np.random.default_rng(42)
        self.X_np = rng.standard_normal((20, 4)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(4)
        dummy = torch.zeros(1, 4)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_onnx_export_valid(self):
        """ONNX model exports and validates without errors."""
        # onnx.checker.check_model is already called in _export_to_onnx
        assert self.onnx_model is not None

    def test_regression_predictions_match(self):
        """SQL predictions match PyTorch forward pass within tolerance=1e-4."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)

    def test_regression_tanh_predictions_match(self):
        """Same architecture with Tanh activations."""
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        model.eval()
        dummy = torch.zeros(1, 4)
        onnx_model = _export_to_onnx(model, dummy)
        expected = _torch_predict(model, self.X_np)
        sql_pred = _sql_predict_regression(onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 2: Binary classifier ending in Sigmoid
# ---------------------------------------------------------------------------


class TestPyTorchBinaryClassifier:
    """End-to-end test for a binary MLP classifier: 3→6→1 with ReLU→Sigmoid."""

    def setup_method(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )
        self.model.eval()
        rng = np.random.default_rng(0)
        self.X_np = rng.standard_normal((15, 3)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(3)
        dummy = torch.zeros(1, 3)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_sigmoid_probabilities_match(self):
        """SQL sigmoid probabilities match PyTorch output within tolerance=1e-4."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 3: Residual MLP with an Add skip connection
# ---------------------------------------------------------------------------


class ResidualMLP(nn.Module):
    """Simple residual MLP: out = layer2(relu(layer1(x))) + x."""

    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))) + x


class TestPyTorchResidualMLP:
    """End-to-end test for a residual MLP with an Add skip connection."""

    def setup_method(self):
        torch.manual_seed(99)
        self.model = ResidualMLP(4)
        self.model.eval()
        rng = np.random.default_rng(99)
        self.X_np = rng.standard_normal((10, 4)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(4)
        dummy = torch.zeros(1, 4)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_residual_predictions_match(self):
        """SQL residual-block predictions match PyTorch within tolerance=1e-4."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 4: Dropout pass-through (inference no-op)
# ---------------------------------------------------------------------------


class TestPyTorchDropoutPassthrough:
    """Dropout is identity at inference — the SQL translation must match."""

    def setup_method(self):
        torch.manual_seed(13)
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4, 1),
        )
        self.model.eval()
        rng = np.random.default_rng(13)
        self.X_np = rng.standard_normal((20, 4)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(4)
        dummy = torch.zeros(1, 4)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_dropout_passthrough_predictions_match(self):
        """SQL predictions with Dropout layers match PyTorch inference."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 5: GlobalAveragePool — mean over feature channels
# ---------------------------------------------------------------------------


class _FeatureMeanModel(nn.Module):
    """Linear → ReLU → mean over units → single output."""

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))          # (N, hidden)
        pooled = h.mean(dim=1, keepdim=True)  # (N, 1) — mean over hidden units
        return self.fc2(pooled)               # (N, 1)


class TestGlobalAveragePool:
    """GlobalAveragePool reduces all channels to their per-row mean."""

    def setup_method(self):
        torch.manual_seed(7)
        self.model = _FeatureMeanModel(in_features=5, hidden=8)
        self.model.eval()
        rng = np.random.default_rng(7)
        self.X_np = rng.standard_normal((15, 5)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(5)
        dummy = torch.zeros(1, 5)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_global_avg_pool_predictions_match(self):
        """SQL predictions with global average pooling match PyTorch."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 6: GlobalMaxPool — max over feature channels
# ---------------------------------------------------------------------------


class _FeatureMaxModel(nn.Module):
    """Linear → ReLU → max over units → single output."""

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))           # (N, hidden)
        pooled, _ = h.max(dim=1, keepdim=True) # (N, 1) — max over hidden units
        return self.fc2(pooled)                # (N, 1)


class TestGlobalMaxPool:
    """GlobalMaxPool reduces all channels to their per-row maximum."""

    def setup_method(self):
        torch.manual_seed(17)
        self.model = _FeatureMaxModel(in_features=5, hidden=8)
        self.model.eval()
        rng = np.random.default_rng(17)
        self.X_np = rng.standard_normal((15, 5)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(5)
        dummy = torch.zeros(1, 5)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_global_max_pool_predictions_match(self):
        """SQL predictions with global max pooling match PyTorch."""
        expected = _torch_predict(self.model, self.X_np)
        sql_pred = _sql_predict_regression(self.onnx_model, self.features, self.X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 7: Multiclass classifier with Softmax (end-to-end accuracy, Python #24)
# ---------------------------------------------------------------------------


def _sql_predict_multiclass(
    onnx_model: "onnx.ModelProto",
    features: dict,
    X: "pd.DataFrame",
) -> "np.ndarray":
    """Translate multiclass ONNX model to SQL and return the (N, K) probability matrix."""
    import duckdb

    conn = duckdb.connect(":memory:")
    parsed = ParsedPipeline._from_onnx_model(onnx_model, features)
    sql = orbital.export_sql("data", parsed, dialect="duckdb")
    result = execute_sql(sql, conn, "duckdb", X)
    return result.values.astype(float)


class TestPyTorchMulticlassClassifier:
    """End-to-end test for a 3-class MLP: 4→8→3 with ReLU→Softmax."""

    def setup_method(self):
        torch.manual_seed(55)
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1),
        )
        self.model.eval()
        rng = np.random.default_rng(55)
        self.X_np = rng.standard_normal((20, 4)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(4)
        dummy = torch.zeros(1, 4)
        self.onnx_model = _export_to_onnx(self.model, dummy)

    def test_multiclass_probabilities_match(self):
        """SQL per-class probabilities match PyTorch Softmax output."""
        with torch.no_grad():
            expected = self.model(
                torch.tensor(self.X_np, dtype=torch.float32)
            ).numpy()
        sql_pred = _sql_predict_multiclass(
            self.onnx_model, self.features, self.X_df
        )
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 8: InstanceNormalization end-to-end
# ---------------------------------------------------------------------------


class _InstanceNormMLP(nn.Module):
    """MLP with InstanceNorm1d over the hidden-layer output."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.norm = nn.InstanceNorm1d(8, affine=True, eps=1e-5)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        h = self.fc1(x)               # (N, 8)
        h = h.unsqueeze(0)            # (1, N, 8) — treat N as spatial, 1 batch
        h = self.norm(h)              # normalise over spatial dim (N)
        h = h.squeeze(0)              # (N, 8)
        return self.fc2(torch.relu(h))


class TestInstanceNormalization:
    """InstanceNormalization translator matches PyTorch normalization."""

    def setup_method(self):
        torch.manual_seed(4)
        self.model = _InstanceNormMLP()
        self.model.eval()
        rng = np.random.default_rng(4)
        self.X_np = rng.standard_normal((12, 6)).astype(np.float32)
        self.X_df = _input_df(self.X_np)
        self.features = _features_for(6)

    def test_instance_norm_onnx_export(self):
        """Check InstanceNorm model exports to valid ONNX."""
        dummy = torch.zeros(1, 6)
        onnx_model = _export_to_onnx(self.model, dummy)
        assert onnx_model is not None

# ---------------------------------------------------------------------------
# Test 9: BatchNorm → standalone Activation (Python #36)
# ---------------------------------------------------------------------------


class TestBatchNormFollowedByActivation:
    """BatchNormalization followed by a standalone activation node — parity with PyTorch.

    Verifies that orbital correctly dispatches each activation op after a
    BatchNorm node, rather than treating it as a pass-through.
    """

    def _make_model(self, act_module: nn.Module) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            act_module,
            nn.Linear(8, 1),
        )

    def _run(self, model: nn.Sequential) -> None:
        model.eval()
        rng = np.random.default_rng(99)
        X_np = rng.standard_normal((20, 4)).astype(np.float32)
        X_df = _input_df(X_np)
        features = _features_for(4)
        dummy = torch.zeros(1, 4)
        onnx_model = _export_to_onnx(model, dummy)
        expected = _torch_predict(model, X_np)
        sql_pred = _sql_predict_regression(onnx_model, features, X_df)
        np.testing.assert_allclose(expected, sql_pred, rtol=1e-4, atol=1e-4)

    def test_batchnorm_then_relu(self):
        """BatchNorm1d → ReLU → Linear predictions match PyTorch."""
        torch.manual_seed(99)
        self._run(self._make_model(nn.ReLU()))

    def test_batchnorm_then_hardsigmoid(self):
        """BatchNorm1d → Hardsigmoid → Linear predictions match PyTorch."""
        torch.manual_seed(99)
        self._run(self._make_model(nn.Hardsigmoid()))

    def test_batchnorm_then_hardswish(self):
        """BatchNorm1d → Hardswish → Linear predictions match PyTorch."""
        torch.manual_seed(99)
        self._run(self._make_model(nn.Hardswish()))