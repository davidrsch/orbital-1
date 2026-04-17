import logging
import sqlite3
import math
import sys

# Pre-import orbital and its heavy C dependencies (onnx, skl2onnx) before
# pytest's assertion-rewriting hook is active.  Without this, onnx's
# dynamic_class_creation() in skl2onnx/algebra/onnx_ops.py triggers a
# Windows access violation when executed through pytest's import machinery.
import orbital  # noqa: F401 – side-effect pre-load
from orbital.translation.optimizer import Optimizer

import duckdb
import sqlalchemy
import numpy as np
import pandas as pd
import pytest
from onnx import helper
from sklearn.datasets import load_diabetes, load_iris

PY39 = sys.version_info[:2] < (3, 10)


# ---------------------------------------------------------------------------
# Shared ONNX graph builder
# ---------------------------------------------------------------------------


def make_graph_with_inits(node, inputs_info, outputs_info, initializers):
    """Create an ONNX GraphProto with the given node, I/O specs, and initializers.

    This helper is used across multiple test modules to build minimal ONNX
    graphs without the boilerplate of ``helper.make_graph``.
    """
    return helper.make_graph(
        [node],
        "test_graph",
        inputs_info,
        outputs_info,
        initializer=initializers,
    )


def pytest_configure(config):
    # Enable debug logging for the projec itself,
    # so that in case of errors during tests we have
    # additional debug information.
    specific_logger = logging.getLogger("orbital")
    specific_logger.setLevel(logging.DEBUG)

    # Use deterministic seed for reproducible test results
    np.random.seed(42)


# Shared fixtures for all test files
@pytest.fixture(scope="class", autouse=True)
def _no_op_optimizer(request):
    """Provide a disabled Optimizer as a class attribute for translator tests."""
    if request.cls is not None:
        request.cls.optimizer = Optimizer(enabled=False)


@pytest.fixture(scope="class")
def iris_data():
    """Load and prepare the iris dataset for testing."""
    iris = load_iris()
    # Clean feature names to match what's used in the example
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = pd.DataFrame(iris.data, columns=feature_names)  # Use clean names directly
    y = pd.DataFrame(iris.target, columns=["target"])
    df = pd.concat([X, y], axis=1)
    return df, feature_names


@pytest.fixture(scope="class")
def diabetes_data():
    """Load and prepare the diabetes dataset for testing."""
    diabetes = load_diabetes()
    feature_names = diabetes.feature_names
    X = pd.DataFrame(diabetes.data, columns=feature_names)
    y = pd.DataFrame(diabetes.target, columns=["target"])
    df = pd.concat([X, y], axis=1)
    return df, feature_names


@pytest.fixture(params=["duckdb", "sqlite", "postgres"])
def db_connection(request):
    """Create database connections for testing SQL exports."""
    dialect = request.param
    if dialect == "duckdb":
        conn = duckdb.connect(":memory:")
        yield conn, dialect
        conn.close()
    elif dialect == "sqlite":
        conn = sqlite3.connect(":memory:")
        if PY39:
            # Python 3.9 sqlite is compiled without -DSQLITE_ENABLE_MATH_FUNCTIONS
            conn.create_function("exp", 1, math.exp)
        yield conn, dialect
        conn.close()
    elif dialect == "postgres":
        try:
            conn = sqlalchemy.create_engine(
                "postgresql://orbitaltestuser:orbitaltestpassword@localhost:5432/orbitaltestdb"
            )
            with conn.connect() as testcon:
                testcon.execute(sqlalchemy.text("SELECT 1"))  # Test connection
        except (sqlalchemy.exc.OperationalError, ImportError):
            pytest.skip("Postgres database not available")
        yield conn, dialect
        conn.dispose()
