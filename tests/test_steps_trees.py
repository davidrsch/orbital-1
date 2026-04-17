"""Tests for ml pipeline step translators."""

import math

import onnx
import ibis
import numpy as np
import pytest
from onnx import TensorProto, helper

from orbital.translate import TRANSLATORS
from orbital.translation.variables import (
    GraphVariables,
    NumericVariablesGroup,
    ValueVariablesGroup,
    VariablesGroup,
)
from orbital.translation.optimizer import Optimizer
from orbital.translation.options import TranslationOptions


from conftest import make_graph_with_inits as _make_graph_with_inits


class TestTreeEnsembleClassifierTranslator:
    def test_binary_classification_single_tree(self):
        """Test TreeEnsembleClassifier with binary classification and single tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import (
            TreeEnsembleClassifierTranslator,
        )

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple binary classification tree:
        # if feature[0] <= 0.5:
        #     return class 0 (weight 1.0)
        # else:
        #     return class 1 (weight 1.0)
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Tree structure attributes
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Class weights
            class_treeids=[0, 0],
            class_nodeids=[1, 2],
            class_ids=[0, 0],
            class_weights=[1.0, -1.0],
            # Class labels
            classlabels_int64s=[0, 1],
            post_transform="NONE",
        )

        # Create a minimal graph and model
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        assert labels == [1, 0, 1]

        # Check probabilities
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "0" in prob_result
        assert "1" in prob_result

    def test_multiclass_classification_single_tree(self):
        """Test TreeEnsembleClassifier with multi-class classification."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import (
            TreeEnsembleClassifierTranslator,
        )

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a multi-class tree with 3 classes
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Simple tree: always returns class weights at leaf 1
            nodes_treeids=[0, 0],
            nodes_nodeids=[0, 1],
            nodes_featureids=[0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF"],
            nodes_values=[10.0, 0.0],
            nodes_truenodeids=[1, 0],
            nodes_falsenodeids=[1, 0],
            nodes_missing_value_tracks_true=[0, 0],
            # Class weights for 3 classes at leaf node
            class_treeids=[0, 0, 0],
            class_nodeids=[1, 1, 1],
            class_ids=[0, 1, 2],
            class_weights=[0.2, 0.5, 0.3],
            # String class labels
            classlabels_strings=["cat", "dog", "bird"],
            post_transform="NONE",
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.STRING, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 3])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        # All inputs should predict "dog" since it has highest weight (0.5)
        assert labels == ["dog", "dog", "dog"]

        # Check probabilities group structure
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "cat" in prob_result
        assert "dog" in prob_result
        assert "bird" in prob_result

    def test_classifier_invalid_input_type(self):
        """Test TreeEnsembleClassifier raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import (
            TreeEnsembleClassifierTranslator,
        )

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            class_treeids=[0],
            class_nodeids=[0],
            class_ids=[0],
            class_weights=[1.0],
            classlabels_int64s=[0, 1],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleClassifier: The first operand must be a column or a column group",
        ):
            translator.process()


class TestTreeEnsembleRegressorTranslator:
    def test_single_tree_regression(self):
        """Test TreeEnsembleRegressor with a single decision tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import (
            TreeEnsembleRegressorTranslator,
        )

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple regression tree:
        # if feature[0] <= 0.5:
        #     return 10.0
        # else:
        #     return 20.0
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Tree structure
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Target weights for regression
            target_treeids=[0, 0],
            target_nodeids=[1, 2],
            target_weights=[10.0, 20.0],
            # Base value (offset)
            base_values=[5.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        predictions = list(backend.execute(result))
        assert predictions == [15.0, 25.0, 15.0]

    def test_regression_base_values_applied(self):
        """Test TreeEnsembleRegressor correctly applies base_values."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import (
            TreeEnsembleRegressorTranslator,
        )

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        # Simple tree that always returns same value
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Single leaf node tree
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            # Weight at leaf
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[7.0],
            # Base value should be added
            base_values=[3.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        # Single-leaf tree returns a constant expression (scalar)
        assert computed == 10.0

    def test_regressor_invalid_input_type(self):
        """Test TreeEnsembleRegressor raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import (
            TreeEnsembleRegressorTranslator,
        )

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[1.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleRegressor: The first operand must be a column or a column group",
        ):
            translator.process()
