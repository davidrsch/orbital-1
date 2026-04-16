"""Defines the translation step for the ArgMin operation."""

import ibis

from ..translator import Translator


class ArgMinTranslator(Translator):
    """Processes an ArgMin node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the ArgMin operation.

    The ArgMin implementation is currently limited to emitting a variable
    that represents the index of the column with the minimum value in a group
    of columns. It is not possible to compute the min of a set of rows
    thus axis must be 1 and keepdims must be 1.

    As it computes the minimum out of a set of columns, argmin expects
    a columns group as its input.

    The limitation is due to the fact that we can't mix variables with
    a different amount of rows and MIN(col) would end up producing a single row.
    This is usually ok, as ArgMin is primarily used to pick the values
    with the minimum value out of the features analyzed by the model,
    and thus it is only required to produce a new value for each entry
    on which to perform a prediction/classification (row).
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__ArgMin.html
        data = self._variables.consume(self.inputs[0])
        axis = self._attributes.get("axis", 0)
        keepdims = self._attributes.get("keepdims", 1)
        select_last_index = self._attributes.get("select_last_index", 0)

        if not isinstance(data, dict):
            # if it's a single column, we can't really do much with it
            # as there aren't other columns to compare with.
            raise NotImplementedError(
                "ArgMinTranslator can only be applied to a group of columns"
            )

        if axis != 1:
            # For axis=0 we would want to return the index of the row
            # with the minimum value, but we don't have a row identifier
            raise NotImplementedError("ArgMinTranslator only supports axis=1")
        if keepdims != 1:
            raise NotImplementedError(
                "ArgMinTranslator only supports retaining original dimensions"
            )

        keys = list(data.keys())
        if len(keys) == 0:
            raise ValueError("ArgMinTranslator requires at least one column")

        if len(keys) == 1:
            self.set_output(data[keys[0]])
            return

        # Generate a CASE THEN ELSE expression to find
        # which out of all the columns has the minimum value.
        # When select_last_index=1, iterate in reverse so that the first
        # column that is <= all others (scanned right-to-left) is the last
        # tied minimum.  For select_last_index=0 iterate forward.
        iteration = (
            list(reversed(list(enumerate(keys))))
            if select_last_index
            else list(enumerate(keys))
        )
        cases: list[tuple[ibis.Expr, int]] = []
        for idx, key in iteration:
            cond = None
            # Compare the current column against every other column.
            for j, other in enumerate(keys):
                if j == idx:
                    continue
                # Use <= so we find the minimum; ties are broken by iteration order.
                cmp_expr = data[key] <= data[other]
                cond = cmp_expr if cond is None else cond & cmp_expr
            cases.append((cond, idx))
        argmin_expr = ibis.cases(*cases, else_=0)

        self.set_output(argmin_expr)
