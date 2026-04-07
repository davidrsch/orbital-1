"""Test individual pipeline steps/translators."""

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


class TestStepCoverage:
    """Verify that all registered steps have corresponding test classes."""

    def test_all_registered_steps_have_tests(self):
        """Every step registered in TRANSLATORS must have a test class.

        Test classes must follow the naming convention Test{OperationName}Translator.
        This test scans both the current module and any ``test_steps_*.py`` sibling
        files so that test classes split into focused sub-modules are also found.
        """
        import importlib.util
        import sys
        from pathlib import Path

        existing_test_classes: set[str] = set()

        # Scan current module
        module = sys.modules[__name__]
        existing_test_classes.update(
            name
            for name in dir(module)
            if name.startswith("Test") and isinstance(getattr(module, name), type)
        )

        # Scan sibling test_steps_*.py modules (future split files)
        test_dir = Path(__file__).parent
        for py_file in sorted(test_dir.glob("test_steps_*.py")):
            mod_name = py_file.stem
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    existing_test_classes.update(
                        name
                        for name in dir(mod)
                        if name.startswith("Test") and isinstance(getattr(mod, name), type)
                    )
                except Exception:
                    pass  # ignore import errors in sibling files

        missing_tests = []
        for operation in sorted(TRANSLATORS.keys()):
            expected_test_class = f"Test{operation}Translator"
            if expected_test_class not in existing_test_classes:
                missing_tests.append((operation, expected_test_class))

        if missing_tests:
            missing_list = "\n".join(
                f"  - {op}: {test_class}" for op, test_class in missing_tests
            )
            pytest.fail(
                f"The following {len(missing_tests)} steps are missing test classes:\n"
                f"{missing_list}\n\n"
                f"Add a test class for each step to test_pipeline_steps.py or a "
                f"test_steps_*.py sibling file."
            )


