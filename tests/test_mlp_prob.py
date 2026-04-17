"""Tests for sklearn MLPClassifier multiclass per-class probability output via ZipMap.

These tests verify that the ZipMap translator correctly maps softmax scores to
per-class probability columns, matching sklearn's predict_proba() output exactly.

This is the Python parity for orbital R issue #19.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
from orbital import types
from orbital.ast import parse_pipeline
from orbital_testing_helpers import execute_sql


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _make_data():
    """Return X (DataFrame), y_bin, and y_multi arrays with a fixed seed."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(60),
            "f2": rng.standard_normal(60),
            "f3": rng.standard_normal(60),
        }
    )
    y_bin = (X["f1"] + X["f2"] > 0).astype(int)
    y_multi = pd.cut(
        X["f1"] * 2 + X["f2"] - X["f3"],
        bins=3,
        labels=[0, 1, 2],
    ).astype(int)
    return X, y_bin, y_multi


FEATURES = {
    "f1": types.DoubleColumnType(),
    "f2": types.DoubleColumnType(),
    "f3": types.DoubleColumnType(),
}


def _validate_proba(pipeline, X, classes):
    """Translate pipeline to SQL, execute against DuckDB, compare to predict_proba."""
    import duckdb

    conn = duckdb.connect(":memory:")
    parsed = parse_pipeline(pipeline, FEATURES)
    sql = orbital.export_sql("data", parsed, dialect="duckdb")
    sql_results = execute_sql(sql, conn, "duckdb", X)
    sklearn_proba = pipeline.predict_proba(X)
    for i, cls in enumerate(classes):
        col_name = f"output_probability.{cls}"
        assert col_name in sql_results.columns, (
            f"Expected column '{col_name}' in SQL output. "
            f"Available columns: {list(sql_results.columns)}"
        )
        np.testing.assert_allclose(
            sql_results[col_name].values.flatten(),
            sklearn_proba[:, i],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Probability mismatch for class {cls}",
        )
    return sql_results


# ---------------------------------------------------------------------------
# Test 1: Binary MLPClassifier — 2 probability columns
# ---------------------------------------------------------------------------


class TestMLPClassifierBinaryProba:
    """Binary MLPClassifier: verify two per-class probability columns are correct."""

    def setup_method(self):
        self.X, self.y_bin, _ = _make_data()
        self.pipeline = Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(self.X, self.y_bin)

    def test_binary_proba_columns_exist(self):
        """Output contains exactly two probability columns for binary classification."""
        import duckdb

        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(self.pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        proba_cols = [c for c in result.columns if c.startswith("output_probability.")]
        assert len(proba_cols) == 2, (
            f"Expected 2 probability columns, got: {proba_cols}"
        )

    def test_binary_proba_values_match(self):
        """Per-class probability values match sklearn predict_proba() within 1e-4."""
        _validate_proba(self.pipeline, self.X, self.pipeline.classes_)

    def test_binary_proba_sums_to_one(self):
        """All per-class probabilities should sum to 1.0 for each row."""
        import duckdb

        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(self.pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        proba_cols = [c for c in result.columns if c.startswith("output_probability.")]
        row_sums = result[proba_cols].sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values,
            np.ones(len(self.X)),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_binary_proba_with_scaler(self):
        """Binary probability output works when StandardScaler precedes the MLP."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        activation="relu",
                        max_iter=500,
                        random_state=7,
                    ),
                ),
            ]
        )
        pipeline.fit(self.X, self.y_bin)
        _validate_proba(pipeline, self.X, pipeline.classes_)


# ---------------------------------------------------------------------------
# Test 2: Multiclass MLPClassifier — N probability columns (3 classes)
# ---------------------------------------------------------------------------


class TestMLPClassifierMulticlassProba:
    """Multiclass MLPClassifier (3 classes): verify N per-class probability columns."""

    def setup_method(self):
        self.X, _, self.y_multi = _make_data()
        self.pipeline = Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(self.X, self.y_multi)

    def test_multiclass_proba_column_count(self):
        """Output contains exactly N=3 probability columns for 3-class problem."""
        import duckdb

        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(self.pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        proba_cols = [c for c in result.columns if c.startswith("output_probability.")]
        n_classes = len(self.pipeline.classes_)
        assert len(proba_cols) == n_classes, (
            f"Expected {n_classes} probability columns, got {len(proba_cols)}: {proba_cols}"
        )

    def test_multiclass_proba_values_match(self):
        """Per-class probability values match sklearn predict_proba() across all 3 classes."""
        _validate_proba(self.pipeline, self.X, self.pipeline.classes_)

    def test_multiclass_proba_sums_to_one(self):
        """Multiclass per-class probabilities sum to 1.0 for each row."""
        import duckdb

        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(self.pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        proba_cols = [c for c in result.columns if c.startswith("output_probability.")]
        row_sums = result[proba_cols].sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values,
            np.ones(len(self.X)),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_multiclass_proba_tanh(self):
        """Multiclass probability works with tanh activation."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(10, 5),
                        activation="tanh",
                        max_iter=1000,
                        random_state=13,
                    ),
                ),
            ]
        )
        pipeline.fit(self.X, self.y_multi)
        _validate_proba(pipeline, self.X, pipeline.classes_)


# ---------------------------------------------------------------------------
# Test 3: Label output still works when probability is not requested
# ---------------------------------------------------------------------------


class TestMLPClassifierLabelOutput:
    """Ensure that label-only (non-proba) output is unaffected by ZipMap support."""

    def setup_method(self):
        self.X, self.y_bin, self.y_multi = _make_data()

    def test_binary_label_output(self):
        """Label output matches sklearn.predict() for binary classification."""
        import duckdb

        pipeline = Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        max_iter=500,
                        random_state=0,
                    ),
                ),
            ]
        )
        pipeline.fit(self.X, self.y_bin)
        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        sql_labels = result["output_label"].astype(int).values
        sklearn_labels = pipeline.predict(self.X)
        np.testing.assert_array_equal(sklearn_labels, sql_labels)

    def test_multiclass_label_output(self):
        """Label output matches sklearn.predict() for multiclass classification."""
        import duckdb

        pipeline = Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        max_iter=1000,
                        random_state=0,
                    ),
                ),
            ]
        )
        pipeline.fit(self.X, self.y_multi)
        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(pipeline, FEATURES)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        result = execute_sql(sql, conn, "duckdb", self.X)
        sql_labels = result["output_label"].astype(int).values
        sklearn_labels = pipeline.predict(self.X)
        np.testing.assert_array_equal(sklearn_labels, sql_labels)
